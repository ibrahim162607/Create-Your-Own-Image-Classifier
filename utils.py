import argparse
import torch
from torchvision import datasets, transforms
import json
import os

def save_model_checkpoint(save_path, model, optimizer, args, classifier):
    '''Save the model checkpoint to the specified path'''
    
    model_checkpoint = {
        'architecture': args.arch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'classifier': classifier,
        'epochs': args.epochs,
        'class_mapping': model.class_to_idx
    }
    
    torch.save(model_checkpoint, save_path)

def load_model_checkpoint(checkpoint_path):
    '''Load the model checkpoint from the specified path'''
    
    loaded_checkpoint = torch.load(checkpoint_path)
    model = loaded_checkpoint['model_state']
    model.classifier = loaded_checkpoint['classifier']
    
    model.load_state_dict(loaded_checkpoint['model_state'])
    model.class_to_idx = loaded_checkpoint['class_mapping']
    
    return model

def load_category_names(json_filepath):
    '''Load category names from a JSON file'''
    
    with open(json_filepath, 'r') as json_file:
        categories = json.load(json_file)
    
    return categories
