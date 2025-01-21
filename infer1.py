import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)
        
    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

def create_confidence_description(probability):
    """Create a text description of confidence level."""
    if probability >= 90:
        return "Very High Confidence"
    elif probability >= 70:
        return "High Confidence"
    elif probability >= 50:
        return "Moderate Confidence"
    elif probability >= 30:
        return "Low Confidence"
    else:
        return "Very Low Confidence"

def process_image(image_path, model, device, transform):
    """Process a single image and return IDC prediction."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Calculate probability
        probability = float(output.item() * 100)
        
        # Get intermediate feature maps (for visualization)
        features = model.get_features(input_tensor)
        
        return {
            'probability': probability,
            'has_idc': probability > 50,  # threshold can be adjusted
            'confidence_level': create_confidence_description(probability),
            'features': features
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def visualize_predictions(image_paths, results, save_dir):
    """
    Create visualization grid showing original images and predictions.
    """
    num_images = len(image_paths)
    images_per_plot = 10
    num_plots = (num_images + images_per_plot - 1) // images_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * images_per_plot
        end_idx = min(start_idx + images_per_plot, num_images)
        current_batch = list(zip(image_paths[start_idx:end_idx], 
                               results[start_idx:end_idx]))
        num_current_images = len(current_batch)
        
        # Create figure
        fig = plt.figure(figsize=(15, 3 * num_current_images))
        
        for idx, (image_path, result) in enumerate(current_batch):
            # Load and display original image
            image = Image.open(image_path).convert('RGB')
            
            plt.subplot(num_current_images, 2, idx * 2 + 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Display prediction results
            plt.subplot(num_current_images, 2, idx * 2 + 2)
            plt.text(0.5, 0.5, 
                    f"IDC: {'Yes' if result['has_idc'] else 'No'}\n" +
                    f"Probability: {result['probability']:.1f}%\n" +
                    f"Confidence: {result['confidence_level']}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    color='red' if result['has_idc'] else 'green')
            plt.axis('off')
            
            # Add filename as text
            plt.figtext(0.01, 1 - (idx + 0.5) / num_current_images,
                       os.path.basename(image_path),
                       fontsize=8)
        
        # Adjust layout and save
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'predictions_batch_{plot_idx + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved visualization batch {plot_idx + 1} to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='IDC Detection Inference using ResNet')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directory to save predictions')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading model...")
    model = ResNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet typically uses 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Get list of images to process
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        print("No valid image files found.")
        return
    
    # Process images
    print(f"\nProcessing {len(image_paths)} images...")
    results = []
    
    for image_path in tqdm(image_paths):
        result = process_image(image_path, model, device, transform)
        if result:
            results.append(result)
    
    # Create visualizations
    visualize_predictions(image_paths, results, args.output_dir)
    
    # Save results to text file
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("IDC Detection Results\n")
        f.write("===================\n\n")
        for image_path, result in zip(image_paths, results):
            f.write(f"File: {os.path.basename(image_path)}\n")
            f.write(f"IDC Prediction: {'Positive' if result['has_idc'] else 'Negative'}\n")
            f.write(f"Probability: {result['probability']:.2f}%\n")
            f.write(f"Confidence Level: {result['confidence_level']}\n")
            f.write("-------------------\n")
    
    print("\nProcessing Complete!")
    print(f"Total images processed: {len(results)}")
    positive_cases = sum(1 for r in results if r['has_idc'])
    print(f"IDC Positive cases: {positive_cases} ({positive_cases/len(results)*100:.1f}%)")
    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()