import argparse
import json
import os
import pathlib

import numpy as np
from numpy.ma import MaskedArray

# MONKEY PATCH FOR SKOPT version incompatibility
import sklearn.utils.fixes

sklearn.utils.fixes.MaskedArray = MaskedArray
from skopt import gbrt_minimize
# END

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow import keras
from tensorflow.keras import Sequential, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from skopt import gp_minimize
from skopt.space import Real, Categorical
from ..wb_attack_layer import WBAttackLayer
from ..wb_attack_loader import WBAttackLoader
from functools import partial
from skopt.utils import use_named_args

assert os.getcwd().endswith("FIA"), "script should be started from home folder"
EARLY_STOPPING_CALLBACK = keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=10,
                                                        restore_best_weights=True)
"""
    This script only runs the MIA training
"""


def get_avg_confidence(true_y, pred_y):
    conf = 0
    for i, y in enumerate(true_y):
        if y == 0:
            conf += 1 - pred_y[i]
        else:
            conf += pred_y[i]
    if len(pred_y) <= 0:
        return 0
    return conf / len(pred_y)


def get_scores(true_y, conf_y):
    pred_y = np.round(conf_y)
    accuracy = accuracy_score(true_y, pred_y)
    precision = precision_score(true_y, pred_y)
    recall = recall_score(true_y, pred_y)
    avg_conf = get_avg_confidence(true_y, conf_y)

    return accuracy, precision, recall, avg_conf


def split_data(Y, train_size, test_size, seed):
    STRAT_SHUFF_SPLIT = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=test_size,
                                               random_state=seed)
    return list(STRAT_SHUFF_SPLIT.split(np.zeros(len(Y)), Y))[0]


# space = [Categorical([2 ** i for i in range(5, 7)], name='batch_size'),
#        Real(5e-5, 1e-3, 'log-uniform', name='learning_rate'),
#        Categorical([False, True], name='batch_norm'),
#        Categorical([1, 2, 4, 8, 16, 32, 64, 100], name='conv')]


space = [Categorical([64], name='batch_size'),
         Real(0.0001, 0.00010000001, 'log-uniform', name='learning_rate'),
         Categorical([True], name='batch_norm'),
         Categorical([1, 2, 4, 8, 16, 32, 64, 100], name='conv')]


def create_model(save_epochs, batch_size, num_classes, layers_used, input_size, learning_rate=0.0001, batch_norm=False,
                 conv_height=4):
    print("params: batch size", batch_size, " learning_rate", learning_rate, " batch norm", batch_norm, " conv height",
          conv_height)
    model = Sequential()
    weight_initializer = RandomNormal(stddev=0.01)
    model.add(Input(shape=(save_epochs, input_size), batch_size=batch_size))
    model.add(WBAttackLayer(layers_used, num_classes, [],
                            0.2, weight_initializer, batch_norm=batch_norm, stacked_epochs=save_epochs,
                            conv_height=conv_height))
    model.add(Dense(256, activation=LeakyReLU(alpha=0.1), kernel_initializer=weight_initializer, name="dense1"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation=LeakyReLU(alpha=0.1), kernel_initializer=weight_initializer))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation=LeakyReLU(alpha=0.1), kernel_initializer=weight_initializer))
    model.add(Dense(1, activation="sigmoid", kernel_initializer=weight_initializer))
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.summary()
    return model


def objective(wb_loader_train, wb_loader_test, save_epochs, num_classes, layers_used, workers, params_to_tune):
    OBJ_EARLY_STOPPING_CALLBACK = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                                restore_best_weights=True)
    wb_loader_test.batch_size = params_to_tune[0]
    wb_loader_train.batch_size = params_to_tune[0]
    model = create_model(save_epochs, params_to_tune[0], num_classes, layers_used,
                         wb_loader_train.get_input_size(),
                         params_to_tune[1], params_to_tune[2], params_to_tune[3])
    model.fit_generator(wb_loader_train, epochs=500, validation_data=wb_loader_test,
                        validation_steps=10, workers=workers,
                        callbacks=[OBJ_EARLY_STOPPING_CALLBACK])
    tmp_true = np.array([1, 0])
    test_indices = np.array(wb_loader_test.get_indices())[:, 1]
    train_indices = np.array(wb_loader_train.get_indices())[:, 1]

    attack_train_size = min(len(train_indices), len(test_indices))
    attack_train_size = (attack_train_size // int(params_to_tune[0])) * int(params_to_tune[0])
    test_conf = np.array(model.predict_generator(wb_loader_test).ravel())
    test_pred = np.round(test_conf)
    true_y_test = np.tile(tmp_true, (attack_train_size // 2))

    return -(accuracy_score(true_y_test, test_pred))


def train(batch_size, data, merged, output, plain_indices, save_epochs, workers, num_classes, optimize=False,
          learning_rate=0.0001):
    """
    Train a model by a merged data file
    """
    print(f"Writing tensorboard logs to {output}/logs/")
    pathlib.Path(f"{output}/logs").mkdir(parents=True, exist_ok=True)
    TENSORBOARD_CALLBACK = keras.callbacks.TensorBoard(log_dir=f"{output}/logs")
    print("loading indices")
    if isinstance(plain_indices, str):
        indices = np.load(plain_indices, allow_pickle=True).item()
        train_indices = indices['train']
        test_indices = indices['test']
    elif isinstance(plain_indices, list):
        train_indices = list(range(len(plain_indices) // 2))
        test_indices = list(range(len(plain_indices) // 2, len(plain_indices)))
    print("loading data")
    data = np.load(data, allow_pickle=True)
    x = data['x']
    y = data['y']
    print("preparing data (train/test split)")
    SPLIT_SIZE = len(train_indices) // 2
    target_train1, target_train2 = list(range(0, SPLIT_SIZE)), list(range(SPLIT_SIZE, len(train_indices)))
    target_test1, target_test2 = list(range(0, SPLIT_SIZE)), list(range(SPLIT_SIZE, len(train_indices)))
    assert len(train_indices) <= len(test_indices), "test set length should be equal or more than train set"
    assert set(target_test1).isdisjoint(target_test2), "test sets are not disjoint"
    assert set(target_train1).isdisjoint(target_train2), "train sets are not disjoint"
    assert set(train_indices).isdisjoint(test_indices), "train set is not disjoint with test set"
    print("create data loader generators")
    file = open(merged, 'r')
    attack_data_inf = json.load(file)
    wb_loader_train = WBAttackLoader(attack_data_inf["attack_train_file"],
                                     attack_data_inf["attack_test_file"],
                                     target_train1, target_test1, batch_size)
    wb_loader_test = WBAttackLoader(attack_data_inf["attack_train_file"],
                                    attack_data_inf["attack_test_file"],
                                    target_train2, target_test2, batch_size, shuffle=False)
    print("compiling encoder model")
    model = create_model(save_epochs, batch_size, num_classes, attack_data_inf['layers_used'],
                         wb_loader_train.get_input_size(), learning_rate=learning_rate, batch_norm=True)
    if optimize == 0:
        print("train MIA WB model")
        model.fit_generator(wb_loader_train, epochs=500, validation_data=wb_loader_test,
                            validation_steps=10, workers=workers,
                            callbacks=[TENSORBOARD_CALLBACK, EARLY_STOPPING_CALLBACK])
        print("saving")
        model.save(f'{output}/attacker_model.h5')
    else:
        print("perform bayesian minimization")
        res_gp = gp_minimize(partial(objective, wb_loader_train, wb_loader_test, save_epochs, num_classes,
                                     attack_data_inf['layers_used'], workers), space, n_calls=20, random_state=42)
        print(res_gp)
        return

    print('Evaluating performance...')
    wb_loader_train = WBAttackLoader(attack_data_inf["attack_train_file"],
                                     attack_data_inf["attack_test_file"],
                                     target_train1, target_test1, batch_size, shuffle=False)
    tmp_true = np.array([1, 0])
    test_indices = np.array(wb_loader_test.get_indices())[:, 1]
    train_indices = np.array(wb_loader_train.get_indices())[:, 1]

    attack_train_size = min(len(train_indices), len(test_indices))
    attack_train_size = (attack_train_size // batch_size) * batch_size
    true_y_train = np.tile(tmp_true, (attack_train_size // 2))
    true_y_test = np.tile(tmp_true, (attack_train_size // 2))
    train_conf = np.array(model.predict_generator(wb_loader_train).ravel())
    train_pred = np.round(train_conf)
    train_classes = y[train_indices]
    train_classes = train_classes.flatten()
    train_classes = train_classes[:attack_train_size]
    train_accuracy, train_precision, train_recall, train_avg_conf = get_scores(true_y_train, train_conf)
    print("Train Accuracy: {}".format(train_accuracy))
    print("Train Precision: {}".format(train_precision))
    print("Train Recall: {}".format(train_recall))
    print("Train Avg Confidence: {}".format(train_avg_conf))
    test_conf = np.array(model.predict_generator(wb_loader_test).ravel())
    test_pred = np.round(test_conf)
    test_classes = y[test_indices]
    test_classes = test_classes.flatten()
    test_classes = test_classes[:attack_train_size]
    test_accuracy, test_precision, test_recall, test_avg_conf = get_scores(true_y_test, test_conf)
    print("Test Accuracy: {}".format(test_accuracy))
    print("Test Precision: {}".format(test_precision))
    print("Test Recall: {}".format(test_recall))
    print("Test Avg Confidence: {}".format(test_avg_conf))
    train_score_per_class = {}
    test_score_per_class = {}
    for c in set(train_classes):
        # train
        train_conf_class = train_conf[np.where(train_classes == c)]
        train_true_y_class = true_y_train[np.where(train_classes == c)]
        train_score_per_class[c] = get_scores(train_true_y_class, train_conf_class)
        # test
    for c in set(test_classes):
        test_conf_class = test_conf[np.where(test_classes == c)]
        test_true_y_class = true_y_test[np.where(test_classes == c)]
        test_score_per_class[c] = get_scores(test_true_y_class, test_conf_class)

    np.savez(output + '/predictions.npz', train_conf=train_conf, test_conf=test_conf, train_pred=train_pred,
             test_pred=test_pred, train_true=true_y_train, test_true=true_y_test, train_classes=train_classes,
             test_classes=test_classes, train_score_per_class=train_score_per_class,
             test_score_per_class=test_score_per_class, test_indices=test_indices, train_indices=train_indices)
    attack_inf = {}
    attack_inf['train_accuracy'] = train_accuracy
    attack_inf['train_precision'] = train_precision
    attack_inf['train_recall'] = train_recall
    attack_inf['train_avg_conf'] = train_avg_conf
    attack_inf['test_accuracy'] = test_accuracy
    attack_inf['test_precision'] = test_precision
    attack_inf['test_recall'] = test_recall
    attack_inf['test_avg_conf'] = test_avg_conf
    attack_inf_file = os.path.join(output + '/final_attack_inf.json')
    with open(attack_inf_file, 'w') as outfile:
        json.dump(attack_inf, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Target model training")
    parser.add_argument("--batch_size", "-b", default=128, type=int, help="Batch Size")
    parser.add_argument("--save_epochs", "-s", default=4, required=True, type=int,
                        help="List of epochs a model should be saved")
    parser.add_argument("--num_classes", "-n", default=100, type=int, required=True, help="Number of labels")
    parser.add_argument("--experiment", "-e", type=str, required=True, help="Experiment name")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output Path")
    parser.add_argument("--merged", "-m", type=str, required=True, help="Path to merged_data_inf.json ")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to data")
    parser.add_argument("--indices", "-i", type=str, required=True, help="Path to indices")
    parser.add_argument("--workers", "-w", default=4, type=int, help="Amount of cpus used to train")

    args = parser.parse_args()
    assert os.path.exists(args.data), "data path should point to a file"
    assert os.path.exists(args.indices), "indices path should point to a file"
    assert os.path.exists(args.data), "data path should exist and point to a .json file"
    assert os.path.isdir(args.output), "output path should be a directory"
    assert os.path.isdir("logs"), "./logs dir should exist"

    args.batch_size = int(args.batch_size)

    train(args.batch_size, args.data, args.merged, args.output,
          args.indices, args.save_epochs, args.workers, args.num_classes, args.experiment)
