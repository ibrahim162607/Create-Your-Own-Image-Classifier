import argparse
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils import load_model_checkpoint, load_category_names

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict image class using a pre-trained model')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--filepath', type=str, default='flowers/test/1/image_06743.jpg', help='Path to image file')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category names file')
    parser.add_argument('--gpu', type=str, default='gpu', help='Use GPU if available')
    return parser.parse_args()

def preprocess_image(image_path):
    img = Image.open(image_path)
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transformations(img)

def make_prediction(image_path, model, top_k=3, device='gpu'):
    model.to(device)
    image_tensor = preprocess_image(image_path).unsqueeze(0).float()

    if device == 'gpu' and torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        output = model(image_tensor)
    
    probabilities = F.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(top_k, dim=1)

    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]

    return top_probs, top_classes

def main():
    args = get_input_args()
    device = 'cuda' if args.gpu == 'gpu' and torch.cuda.is_available() else 'cpu'
    model = load_model_checkpoint(args.checkpoint)
    category_names = load_category_names(args.category_names)
    
    probabilities, classes = make_prediction(args.filepath, model, args.top_k, device)
    class_labels = [category_names[str(cls)] for cls in classes]

    print(f"Predictions for image '{args.filepath}':")
    for label, prob in zip(class_labels, probabilities):
        print(f"{label}: {prob:.4f}")

if __name__ == "__main__":
    main()
