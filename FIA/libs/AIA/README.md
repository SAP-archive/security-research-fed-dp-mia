# Attribute Inference Attack

here are three attribute inference attack models. Only the _sophisticate.py_ is used. 
Yeom et al. and naive implementation yield worse results.

## Sophisticate model

This attribute inference model works like that:

- take the target model data set
- mirror that dataaset and flip the bit of the sensitive attribute
- put the orig_dataset and wrong_dataset into the MIA model
