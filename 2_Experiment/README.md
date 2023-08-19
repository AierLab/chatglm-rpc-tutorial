# Tutorial 2: Enhanced Experimentation Features

`Tutorial 2` extends upon `tutorial 1` by introducing additional functions designed for experimentation. 
Here are the detailed descriptions of these added features:

## 1. Freezing Specific Layers

Users can freeze the weights of specific layers in the model by modifying the `training_args.layers_to_freeze` variable. 
Refer to `main.py`, line 61.

**Format:** The variable is a list containing the indices of the layers to be frozen. 

For example: 
[1, 10, 22] implies that `model.transformer.layers[0]`, `model.transformer.layers[10]`, and `model.transformer.layers[22]` will be frozen before the training starts.

## 2. Dynamically Freezing Layers After a Specific Epoch

This can be achieved by modifying `training_args.dynamic_freeze`. 
Refer to `main.py`, line 62.

**Format:** The variable is a dictionary. The key of the dictionary is the epoch number, and the value is a list containing the indices of the layers to be frozen at that epoch.

For example:
{5: [10, 11, 12], 
10: [13, 15]} 
means that at the beginning of the fifth epoch, layers 10, 11, and 12 will be frozen. At the beginning of the tenth epoch, layers 13 and 15 will be frozen.

## 3. Saving Input and Output Tensors of Specific Layers During Forward Process

Implement this feature by modifying `training_args.hook_layers`. 
Refer to `main.py`, line 64.

**Format:** The variable is a list containing the indices of the model layers for which the input and output values need to be recorded.

For example:
[4, 10, 27] indicates that the input and output of layers 4, 10, and 27 will be saved to the `./intermediate/` directory during the forward process. The saved format is `.pt` (PyTorch tensor files).






