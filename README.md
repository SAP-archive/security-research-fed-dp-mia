# Comparing local and central differential privacy in federated learning using membership inference attacks

[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-fed-dp-mia)](https://api.reuse.software/info/github.com/SAP-samples/security-research-fed-dp-mia)


## Description

SAP Security Research sample code to reproduce research that builds on our paper 
"Comparing local and central differential privacy using membership inference attacks"[1].

## Requirements

- Python 3.6
- h5py
- numpy
- scikit-learn
- scipy
- Tensorflow
- Tensorflow Privacy
- pytest
- Flask
- socketIO
- Scikit-Optimize
- matplotlib


## Download and Installation

### Federated Learning Inference Attack Framework

Implementation of a federated learning framework for inference attacks.

### Install

Running `make install` in the project folder should be enough for most usecases.

It creates the basic project directory structure and installs FIA as well as other requirements. You can use pip as your package manager and install the dpa package via python -m pip install -e ./ For other package managers you need to install dpa using setup.py.

### Directory Structure
After having run `make install`, the following directory structure should be created in your local file system. Note: Everything that must not be tracked by git is already in `.gitignore`.
```

FIA/
    |--Makefile
    |--setup.py
    |--README.md
    |--requirements.txt
    |--.gitignore
    |--data/ 		# Data folder
    |    |--your data files
    |--run_scripes/ # Scripts to orchestrate the experiments
         |--cdp/  
            |--global_attack/ # Global + CDP Experiment
            |--local_attack/  # Local + CDP Experiment
         |--ldp/
            |--global_attack/ # Global + LDP Experiment
            |--local_attack/ # Local + LDP Experiment
         |--nodp/
            |--global_attack/ # NoDP Experiment
            |--local_attack/ # NoDP Experiment 
    |--experiments/
    |    |--your experiments data
    |
    |--logs/	
    |    |--your log files
    |
    |--models/
    |    |--relevant models that should not be trained again
    |
    |--reports/
    |    |--Reports summarizing experiment results
    |--FIA/
        |--libs/			# Source root
            |--FederatedFramework/		# The framework for federated learning
            |--MIA/		# The framework for white box membership inference attacks
            |--AIA/ # The framework for white box attribute inference attacks
```

For every experiment, a folder with the same name as the experiment is created in the experiments folder.

### Datasets

The framework was evaluated with the dataset Purchases Shopping Carts, Texas Hospital Stays and Faces in the Wild.
All LDP perturbed data have to be present in the `./data` folder before running the experiments.
Since we're using VGG for the LFW dataset, a pretrained VGG-Model is expected to reside in the `/models` folder prior to the experiment run:
`./models/rcmalli_vggface_tf_notop_vgg16.h5`.

## Usage

The experiments were run on a fixed set of AWS EC2 instances. The easiest way therefore, is to use them aswell.
We used the `c5d.9xlarge` instance type for Purchases and Texas dataset and `c5d.24xlarge` for the LFW dataset.
Please stick to the `Deep Learning Ubuntu` AMI with the default username `Ubuntu` and the default home-path `/home/ubuntu`, such that the scripts need minimal adjustments.
 

### Script Structure

There are six different `federated_learning.py` scripts. Each of them has a set of parameters at the script header. 
However, all of those have some fixed parameter set in common, namely:
```
ec2_instances = ["ec2-xy-xyz..."]  # list of EC2 Instances
key = "/~/.ssh/aws.pem"  # path to AWS Key
clients = 4  # number of clients training in parallel
target_lr = 0.001  # target model learning rate
target_b = 64  # target model batch size
attack_b = 32  # attack model batch size
attack_lr = 0.0007  # attack model learning rate
```

And some more experiment specific, e.g. the CDP and LDP noise configurations.
These scripts, given a list of EC2 instances, will rsync the folder onto the servers and install the FIA module.
Then start a Federated Learning Aggregator process and several client processes. The training of the target model will the proceed.
Eventually, the MI training will be started given the extracted data from the target-model training step.

### Extend more models

Instead of just Purchases, Texas and LFW, the framework is easy extensible. Simply add another folder to:
`FIA/libs/FederatedFramework/core/experiments` or copy one of the existing. The new folder needs a python file with the same name.
The python file should define the template of the global model, the optimizer to be used for each client and where to find the data and how to split the data.

### Optimize

If the optimize parameter in a `federated_learning.py`-script is set to *1*, no attack model will be created. Instead, some hyperparameter spaces will be evaluated
using Gaussian Process Optimization. The space parameters can be changed in:
```
libs/MIA/experiments/train_wb.py 
```

After execution check the MIA.logs

### Inference

If inference=wb, the classic Nasr et al. white-box membership inference attack is started.
If inference=ai, the attribute inference attack is started

If chosen attribute inference, the optional parameter "--index" can define a list of attributes like:

```
--index range(0,100)
```
This will generate 100 attribute attack models

```
libs/AIA/sophisticate.py
```
There are two other AIA models "naive" and "yeom", but they have an overall inferior performance.

### Results

All results will be written to `./experiments/local` or `./experiments/global`
Then a experiment-specific subfolder will be created by following schema:

`experiment_name = f"{model}{output_size}_cdp_{noise_str}_s_{seed}"`

After a successful run you will find the training- and test-data, and three files containing the results:

1. `training_information.json`
This will contain all information about the FL-Training run: 
training accuracy, validation accuracy, epochs per round, clients, batch size
2. `final_attack_inf.json`
This will contain all information about the WB MIA or AIA model:
    - train accuracy, recall, precision, average confidence
    - test accuracy, recall, precision, average confidence
3. `predictions.npz`
This is a single numpy dump containing:
- Confidence for all training samples
- Confidence for all test samples
- Predictions for all training `samples
- Predictions for all test samples
- True labels for all train samples
- True labels for all test samples
- And some more


## Tests
For automatic testing of the frameworks please use make:

```
make test
```

## Contributors

- Tom Ganz
- Daniel Bernau
- Philip-William Grassal

## Contributing
If you wish to contribute code, offer fixes or improvements, please send a pull request. Due to legal reasons, contributors will be asked to accept a DCO when they create the first pull request to this project. This happens in an automated fashion during the submission process. SAP uses [the standard DCO text of the Linux Foundation](https://developercertificate.org/).

## License
Copyright (c) 2022 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
