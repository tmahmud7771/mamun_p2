import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

# Copy the model architecture classes from training
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=48, patch_size=8, in_channels=3, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        
        x = (attention @ v).transpose(1, 2).reshape(B, N, C)
        x = self.projection(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim, dropout)
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size=48, patch_size=8, in_channels=3, embed_dim=192,
                 depth=12, num_heads=8, mlp_ratio=4.0, dropout=0.1, num_classes=1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
              for _ in range(depth)]
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return torch.sigmoid(x)

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
        
        # Calculate probability
        probability = float(output.item() * 100)
        
        return {
            'probability': probability,
            'has_idc': probability > 50,
            'confidence_level': create_confidence_description(probability),
            'original_image': image
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def visualize_predictions(results, save_dir):
    """Create visualization grid showing predictions."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Process 10 images per figure
    num_images = len(results)
    images_per_plot = 10
    num_plots = (num_images + images_per_plot - 1) // images_per_plot
    
    for plot_idx in range(num_plots):
        start_idx = plot_idx * images_per_plot
        end_idx = min(start_idx + images_per_plot, num_images)
        current_batch = list(results.items())[start_idx:end_idx]
        
        # Create figure
        fig = plt.figure(figsize=(15, 3 * len(current_batch)))
        
        for idx, (image_path, result) in enumerate(current_batch):
            # Display original image
            plt.subplot(len(current_batch), 2, 2*idx + 1)
            plt.imshow(result['original_image'])
            plt.title('Original Image')
            plt.axis('off')
            
            # Display prediction
            plt.subplot(len(current_batch), 2, 2*idx + 2)
            color = 'red' if result['has_idc'] else 'green'
            plt.text(0.5, 0.5,
                    f"IDC: {'Positive' if result['has_idc'] else 'Negative'}\n" +
                    f"Probability: {result['probability']:.1f}%\n" +
                    f"Confidence: {result['confidence_level']}",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes,
                    color=color,
                    fontsize=12)
            plt.axis('off')
            
            # Add filename
            plt.figtext(0.01, 1 - (idx + 0.5)/len(current_batch),
                       os.path.basename(image_path),
                       fontsize=8)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'predictions_batch_{plot_idx+1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='IDC Detection using Vision Transformer')
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
    model = VisionTransformer().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
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
            f.write(f"IDC Detection: {'Positive' if result['has_idc'] else 'Negative'}\n")
            f.write(f"Probability: {result['probability']:.2f}%\n")
            f.write(f"Confidence Level: {result['confidence_level']}\n")
            f.write("-------------------\n")
    
    # Print summary
    print("\nProcessing Complete!")
    print(f"Total images processed: {len(results)}")
    positive_cases = sum(1 for r in results.values() if r['has_idc'])
    print(f"IDC Positive cases: {positive_cases} ({positive_cases/len(results)*100:.1f}%)")
    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()