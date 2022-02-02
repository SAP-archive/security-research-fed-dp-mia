# MIA Experiments

Run scripts for federated MIA


The workflow is as follows:
1) extract_features.py: generates for each Epoch-Model the MIA WB features
2) merge_features_py: merges the attack features per epoch together
3) train_wb.py: trains the mia wb model and saves it

*train.py* does everything together. In the most cases just run train.py. Only run the other scripts, if train.py got interrupted e.g. due to a bug/crash/IO-error....
