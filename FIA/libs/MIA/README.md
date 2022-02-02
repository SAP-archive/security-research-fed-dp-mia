# Whitebox MIA

Whitebox MIA is a Membership Inference Attack Framework, hence the name.

## Structure

It consists of three files:
1. Data Generator:  
From a Target model we need to extract
 - Gradient matrix w.r.t. to the inputs
 - Loss
 - Label
 - Layer Outputs
 
 The Data Generator takes a model, its Train and Test files and the corresponding indices to create the attack information. 
 Afterwards it serializes a configuration json-file and the Train/Test-Data for the Attacker model.
 
 2. Data Loader:  
 After Generating the data, it is neccessary to deserialize the attack data. The Attack Loader Library can be easily used as
 Data Generator for keras.__model__.fit_generator()
 
 3. Attack Layer:  
 The Attack Layer is a custom Keras layer. It is based upon the paper from Nasr et al. regarding Whitebox Membership Inference Attacks