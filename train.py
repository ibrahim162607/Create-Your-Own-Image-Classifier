import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from utils import save_model_checkpoint, load_model_checkpoint
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def get_input_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model")
    
    # Command line arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--arch', type=str, default='densenet121', choices=['vgg13', 'densenet121'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gpu', type=str, default='gpu', help='Use GPU for training')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Checkpoint save directory')
    
    return parser.parse_args()

def train_model(model, criterion, optimizer, dataloaders, epochs, device):
    output_appear = 10
    steps = 0
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for step, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1
            
            if gpu == 'gpu':
                model.cuda()
                inputs, labels = inputs.to(device), labels.to(device)
            else:
                model.cpu()
                optimizer.zero_grad()
            
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
            
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            if steps % output_appear == 0:
                validate_model(model, criterion, dataloaders, device, running_loss, epoch, epochs, output_appear)
                running_loss = 0

def validate_model(model, criterion, dataloaders, device, running_loss, epoch, epochs, output_appear):
    model.eval()
    validation_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            
            # Calculate accuracy
            ps = torch.exp(outputs)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    print(f"Epoch {epoch+1}/{epochs}.. "
        f"Train Loss: {running_loss/output_appear:.3f}.. "
        f"Validation Loss: {validation_loss/len(dataloaders['val']):.3f}.. "
        f"Validation Accuracy: {accuracy/len(dataloaders['val']):.3f}")
    
    model.train()

def main():
    args = get_input_args()
    
    # Define dataset directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
# Define transforms for training, validation, and testing datasets
    train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Load datasets using ImageFolder and apply respective transforms
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform=val_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Create DataLoader objects for batching and shuffling
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    image_datasets = [train_dataset,val_dataset,test_dataset]
# Store loaders in a dictionary for convenient access
    dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
    }

    # Load pre-trained model
    model = getattr(models, args.arch)(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define the classifier
    if args.arch == "vgg13":
        input_size = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(input_size, 1024)),
                                ('drop', nn.Dropout(p=0.5)),
                                ('relu', nn.ReLU()),
                                ('fc2', nn.Linear(1024, 102)),
                                ('output', nn.LogSoftmax(dim=1))]))
    elif args.arch == "densenet121":
        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024, 500)),('dropout', nn.Dropout(p=0.6)),('relu1', nn.ReLU()),
                        ('fc2', nn.Linear(500, 102)),('output', nn.LogSoftmax(dim=1))]))
    
    model.classifier = classifier
    
    # Set criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    epochs = int(args.epochs)
    # Set device (GPU or CPU)
    device = torch.device("cuda" if args.gpu == 'gpu' and torch.cuda.is_available() else "cpu")
    
    # Train the model
    train_model(model, criterion, optimizer, dataloaders, args.epochs, device)
    
    # Save the model checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    save_model_checkpoint(args.save_dir, model, optimizer, args, classifier)

if __name__ == "__main__":
    main()
