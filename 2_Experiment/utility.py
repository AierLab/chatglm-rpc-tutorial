import torch
import torch.nn as nn
import os
from functools import partial

def freeze_layers(model, layers_to_freeze):
    
    """
    Freeze the specified layers of a PyTorch model.

    Parameters:
    model (nn.Module): The PyTorch model.
    layers_to_freeze (list of int): A list of layer indices that you want to freeze.

    """
    # Convert the children of the model to a list
    layers = list(model.transformer.layers.children())
    
    for index in layers_to_freeze:
        for param in layers[index].parameters():
            param.requires_grad = False
            
def dynamic_freeze(model, freeze_schedule: dict, epoch: int):
    
    """
    Freeze the specified layers after n epochs.

    Parameters:
    model : The PyTorch model.
    freeze_schedule : A dictionary of layer indices that you want to freeze.

    """
    if epoch in freeze_schedule.keys():
        freeze_layers(model, freeze_schedule[epoch])
        
        
def save_input_output(model, module, input, output):
    
    """
    A hook function that saves the input and output of a layer during forward passes.

    Parameters:
    module (nn.Module): The layer
    input (tuple): The input to the layer
    output (tensor): The output from the layer
    """
    
    module.input_tensor = input[0].clone().detach()
    module.output_tensor = output[0].clone().detach() 
    
    save_dir = './intermediates/'
    
    # Check if the directory exists; if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    layer_name = get_layer_name(model, module)
    torch.save(module.input_tensor, os.path.join(save_dir, f'input_{layer_name}.pt'))
    torch.save(module.output_tensor, os.path.join(save_dir, f'output_{layer_name}.pt'))

def register_hooks(model, layer_indices):
    
    """
    Register hooks for the specified layers of a model.

    Parameters:
    model (nn.Module): The PyTorch model.
    layer_indices (list of int): A list of layer indices that you want to register hooks for.

    """
    
    # Convert the children of the model to a list
    layers = list(model.transformer.layers.children())
    
    handles = []
    hook_function = partial(save_input_output, model)
    for index in layer_indices:
        handles.append(layers[index].register_forward_hook(hook_function))
    
    
def get_layer_name(model, layer):
    for name, module in model.named_modules():
        if module is layer:
            return name
    return "Na"