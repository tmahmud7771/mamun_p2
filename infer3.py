import torch
import torch.nn as nn
import clip
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, 2)
        
    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        output = self.classifier(image_features)
        return output

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

def process_image(image_path, model, transform, device):
    """Process a single image and return predictions."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        return {
            'image': image,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_level': create_confidence_description(confidence),
            'probabilities': probabilities[0].cpu().numpy()
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def visualize_predictions(results, save_dir):
    """Create visualization grid showing predictions."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Process images in batches of 10
    batch_size = 10
    for batch_idx in range(0, len(results), batch_size):
        batch_results = list(results.items())[batch_idx:batch_idx + batch_size]
        n_images = len(batch_results)
        
        # Create figure
        fig = plt.figure(figsize=(15, 3 * n_images))
        plt.suptitle('IDC Detection Results', fontsize=16, y=0.95)
        
        for idx, (image_path, result) in enumerate(batch_results):
            # Original image
            plt.subplot(n_images, 2, 2*idx + 1)
            plt.imshow(result['image'])
            plt.title('Original Image')
            plt.axis('off')
            
            # Prediction visualization
            plt.subplot(n_images, 2, 2*idx + 2)
            plt.axis('off')
            
            # Create prediction text
            prediction_text = (
                f"Prediction: {'IDC' if result['predicted_class'] == 1 else 'Non-IDC'}\n"
                f"Confidence: {result['confidence']:.1f}%\n"
                f"Level: {result['confidence_level']}\n"
                f"IDC Prob: {result['probabilities'][1]*100:.1f}%\n"
                f"Non-IDC Prob: {result['probabilities'][0]*100:.1f}%"
            )
            
            # Set color based on prediction
            color = 'red' if result['predicted_class'] == 1 else 'green'
            
            plt.text(0.5, 0.5, prediction_text,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    color=color,
                    fontsize=12)
            
            # Add filename
            plt.figtext(0.01, 1 - (idx + 0.5)/n_images,
                       os.path.basename(image_path),
                       fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'predictions_batch_{batch_idx//batch_size + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='IDC Detection using CLIP-based model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='predictions',
                       help='Directory to save predictions')
    parser.add_argument('--confidence_threshold', type=float, default=50.0,
                       help='Confidence threshold for positive prediction')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, _ = clip.load('ViT-B/32', device=device)
    
    # Initialize model
    print("Loading trained classifier...")
    model = CLIPClassifier(clip_model).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                           (0.26862954, 0.26130258, 0.27577711))
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
    results = {}
    
    for image_path in tqdm(image_paths):
        result = process_image(image_path, model, transform, device)
        if result:
            results[image_path] = result
    
    # Create visualizations
    visualize_predictions(results, args.output_dir)
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write("IDC Detection Results\n")
        f.write("===================\n\n")
        for image_path, result in results.items():
            f.write(f"File: {os.path.basename(image_path)}\n")
            f.write(f"Prediction: {'IDC' if result['predicted_class'] == 1 else 'Non-IDC'}\n")
            f.write(f"Confidence: {result['confidence']:.2f}%\n")
            f.write(f"Confidence Level: {result['confidence_level']}\n")
            f.write(f"IDC Probability: {result['probabilities'][1]*100:.2f}%\n")
            f.write(f"Non-IDC Probability: {result['probabilities'][0]*100:.2f}%\n")
            f.write("-------------------\n")
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Total images processed: {len(results)}")
    positive_cases = sum(1 for r in results.values() if r['predicted_class'] == 1)
    print(f"IDC Positive cases: {positive_cases} ({positive_cases/len(results)*100:.1f}%)")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()