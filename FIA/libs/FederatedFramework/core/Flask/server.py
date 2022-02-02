import codecs
import json
import pickle
import random
import uuid
from threading import Lock, currentThread
from time import sleep
import gc
from flask import *
from flask_socketio import *
import logging
from logging import Handler


class LogHandler(Handler):
    def emit(self, record):
        msg = record.msg % record.args
        if len(msg) < 1000:
            print(f"LOG: {msg}")


LOGGER = logging.getLogger("LOG")
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(LogHandler())


class FLServer(object):
    """
    Federated Averaging algorithm with the server pulling from clients
    The Server synchronizes and instructs the Clients
    """
    LOSS_EPS = .0001  # used for convergence
    UNRESPONSIVE_CLIENT_TOLERANCE = .9
    EARLY_STOPPING_TOLERANCE = 5
    MAX_NUM_ROUNDS = 200
    ROUNDS_BETWEEN_VALIDATIONS = 1
    MIN_ROUNDS = 5

    def __init__(self, global_model, host, port, datasource, indices, parallel_workers=2, min_connected=2, path="./",
                 saved_epochs=None, batch_size=128, epoch_per_round=1):
        """
        Initialize global model
        :param global_model:
        :param host:
        :param port:
        """
        assert min_connected >= parallel_workers, \
            "min_connected must be greater or equal than number of parallel workers!"
        if saved_epochs is not None and int(saved_epochs) > FLServer.MIN_ROUNDS:
            FLServer.MIN_ROUNDS = int(saved_epochs)
        self.ready_client_sids = set()
        self.global_model = global_model()
        self.app = Flask(__name__)
        self.epoch_per_round = epoch_per_round
        self.indices = indices
        self.batch_size = batch_size
        self.host = host
        self.port = port
        self.parallel_workers = parallel_workers
        self.model_id = str(uuid.uuid4())
        self.min_connected = min_connected
        self.socketio = None
        self.X = []
        self.y = []
        self.stopped = False
        self.early_stopping_triggered = 0

        self.datasource = datasource
        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        self.client_rounds = {}
        #####

        ### adversary config
        self.path = path
        self.saved_epochs = saved_epochs

        ###

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):
        """
        Register the async events
        :return:
        """
        self.socketio.on_event('connect', self.handle_connect)
        self.socketio.on_event('disconnect', self.handle_disconnect)
        self.socketio.on_event('reconnect', self.handle_reconnect)
        self.socketio.on_event('client_wake_up', self.handle_wake_up)
        self.socketio.on_event('client_ready', self.handle_client_ready)
        self.socketio.on_event('client_update', self.handle_client_update)
        self.socketio.on_event('client_eval', self.handle_client_eval)

    def handle_connect(self):
        print(request.sid, "connected")

    def handle_reconnect(self):
        print(request.sid, "reconnected")

    def handle_disconnect(self):
        print(request.sid, "disconnected")
        if request.sid in self.ready_client_sids:
            self.ready_client_sids.remove(request.sid)

    def handle_wake_up(self):
        """
        If Client connected and his data is loaded:
        Send him our global model configuration
        :return:
        """
        print("client wake_up: ", request.sid)
        emit('init', {
            'model_json': self.global_model.model.to_json(),
            'model_id': self.model_id,
            'data_split': 0.5,  # test size
            'epoch_per_round': self.epoch_per_round,
            'batch_size': self.batch_size,
            'client_id': request.sid
        })

    def handle_client_ready(self, data):
        """
        If the client loaded his local model
        check if there are enough clients ready and tell them to train one round
        Important: the events are single-threaded and not-reentrant so no need to lock
        :param data:
        :return:
        """
        print("client ready for training", request.sid)
        self.ready_client_sids.add(request.sid)
        self.client_rounds[request.sid] = 0
        if len(self.ready_client_sids) >= self.min_connected:
            self.train_next_round()

    def handle_client_update(self, data):
        """
        On gathered update , average the weights and evaluations.
        The Evaluation follows two ways:
            1. The Data Owners Validation Accuracy and Loss are averaged by the amount of responsive clients (=Training Loss/Accuracy)
            2. The Validation Accuracy and Loss of the global model (=Validation Loss/Accuracy)
        :param data:
        :return:
        """

        print("handle client_update", request.sid, " thread ", currentThread().name)

        print("client updates so far:", len(self.current_round_client_updates) + 1)
        if data['round_number'] != self.current_round or len(
                self.current_round_client_updates) >= self.parallel_workers:
            return

        data["weights"] = pickle_string_to_obj(data['weights'])
        self.current_round_client_updates.append(data)
        snapshot_current_round_client_updates = self.current_round_client_updates.copy()

        if not self.saved_epochs is None and int(data['round_number']) % self.saved_epochs == 0:
            self.global_model.save_client_model(self.path, request.sid, data['round_number'],
                                                self.current_round_client_updates[-1]['weights'])

        if len(snapshot_current_round_client_updates) == \
                self.parallel_workers:
            self.global_model.update_weights(
                [x['weights'] for x in self.current_round_client_updates],
                [float(x['train_size']) for x in self.current_round_client_updates],
            )

            if 'train_loss' in self.current_round_client_updates[0]:
                aggr_train_loss, aggr_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                    [float(x['train_loss']) for x in self.current_round_client_updates],
                    [float(x['train_accuracy']) for x in self.current_round_client_updates],
                    [float(x['train_size']) for x in self.current_round_client_updates],
                    self.current_round
                )
                self.global_model.prev_train_loss = aggr_train_loss

            if 'valid_loss' in self.current_round_client_updates[0]:
                self.global_model.aggregate_test_loss_accuracy(
                    [float(x['valid_accuracy']) for x in self.current_round_client_updates],
                    [float(x['valid_size']) for x in self.current_round_client_updates],
                    [float(x['valid_loss']) for x in self.current_round_client_updates],
                    self.current_round
                )

            aggr_valid_loss, aggr_valid_accuracy = self.global_model.evaluate(self.X[self.indices],
                                                                              self.y[self.indices])
            # Early Stopping
            if data['round_number'] > FLServer.MIN_ROUNDS and data[
                'round_number'] > self.global_model.prev_round and \
                    (self.global_model.prev_valid_loss - aggr_valid_loss) < FLServer.LOSS_EPS:
                # converges
                if self.early_stopping_triggered >= FLServer.EARLY_STOPPING_TOLERANCE:
                    print("converges! starting test phase..")
                    self.stop_and_eval()
                    return
                else:
                    self.early_stopping_triggered += 1

            self.global_model.aggregate_valid_loss_accuracy(aggr_valid_loss, aggr_valid_accuracy,
                                                            self.current_round)
            self.global_model.prev_valid_loss = aggr_valid_loss
            self.global_model.prev_round = data['round_number']
            self.client_rounds[request.sid] = self.client_rounds[request.sid] + 1
            if self.current_round >= FLServer.MAX_NUM_ROUNDS:
                print("max round num reached")
                self.stop_and_eval()
            else:
                self.train_next_round()
        else:
            return

    def handle_client_eval(self, data):
        """
        aggregate client evaluations
        """
        if self.eval_client_updates is None:
            return
        self.eval_client_updates.append(data)

        # tolerate unresponsive clients
        if len(self.eval_client_updates) > self.parallel_workers * FLServer.UNRESPONSIVE_CLIENT_TOLERANCE:
            self.global_model.aggregate_loss_accuracy(
                [float(x['test_loss']) for x in self.eval_client_updates],
                [float(x['test_accuracy']) for x in self.eval_client_updates],
                [x['test_size'] for x in self.eval_client_updates],
            )
            print("== done ==")
            self.global_model.save_validation_indices(self.path, self.indices)
            self.eval_client_updates = None  # special value, forbid evaling again
            if self.stopped:
                self.socketio.stop()

    def train_next_round(self):
        """
        Tell the clients to train one round
        # Note: we assume that during training the #workers will be >= min_connected
        """
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []
        print("trigger garbage collector")
        gc.collect()
        print("### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), self.parallel_workers)
        print("request updates from", client_sids_selected)
        self.socketio.sleep(5)
        # by default each client cnn is in its own "room"
        # for rid in client_sids_selected:
        for rid in client_sids_selected:
            print("sending to ", rid)
            self.socketio.sleep(random.randint(1, 2))
            emit('request_update', {
                'model_id': self.model_id,
                'round_number': self.current_round,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
            }, room=rid)

    def stop_and_eval(self):
        """
        tell the clients to stop and evaluate
        """
        self.stopped = True
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                'model_id': self.model_id,
                'current_weights': obj_to_pickle_string(self.global_model.current_weights),
            }, room=rid)
        print("create training_information json for e.g. global attacker model")
        with open(f'{self.path}/training_information.json', 'w') as outfile:
            json.dump({
                "clients": list(self.ready_client_sids),
                "E": self.epoch_per_round,
                "C": self.parallel_workers,
                "B": self.batch_size,
                "validation_accuracy": self.global_model.valid_accuracies,
                "train_accuracy": self.global_model.train_accuracies,
                "save_epochs": self.saved_epochs,
            }, outfile, indent=4, sort_keys=True)

    def start(self):
        """
        start the socket server
        """
        self.X, self.y = self.datasource.get_data()
        # socket io messages
        self.socketio = SocketIO(self.app, logger=LOGGER,
                                 engineio_logger=LOGGER, manage_session=True, ping_interval=100000,
                                 ping_timeout=10000000,
                                 http_compression=False, async_handlers=False, max_http_buffer_size=10000000000,
                                 allow_upgrades=False)
        self.register_handles()
        self.socketio.run(self.app, host="0.0.0.0", port=self.port)


def obj_to_pickle_string(x):
    """
    Serializes objects to strings
    TODO: test other encoding mechanisms
    """
    return codecs.encode(pickle.dumps(x), "base64").decode()


def pickle_string_to_obj(s):
    """
    Serializes objects to strings
    TODO: test other encoding mechanisms
    """
    return pickle.loads(codecs.decode(s.encode(), "base64"))
