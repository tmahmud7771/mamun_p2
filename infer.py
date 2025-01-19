import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = DoubleConv(3, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv3 = DoubleConv(512 + 256, 256)
        self.up_conv2 = DoubleConv(256 + 128, 128)
        self.up_conv1 = DoubleConv(128 + 64, 64)

        # Final conv
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoding path
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)

        conv4 = self.conv4(pool3)

        # Decoding path
        up3 = self.upsample(conv4)
        up3 = torch.cat([up3, conv3], dim=1)
        up_conv3 = self.up_conv3(up3)

        up2 = self.upsample(up_conv3)
        up2 = torch.cat([up2, conv2], dim=1)
        up_conv2 = self.up_conv2(up2)

        up1 = self.upsample(up_conv2)
        up1 = torch.cat([up1, conv1], dim=1)
        up_conv1 = self.up_conv1(up1)

        out = self.final_conv(up_conv1)

        return torch.sigmoid(out)


def process_image(image_path, model, device, transform):
    """Process a single image and return IDC prediction."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)

        # Calculate IDC probability
        prob_map = output.squeeze().cpu().numpy()
        idc_probability = float(np.mean(prob_map > 0.5) * 100)

        return {
            'probability': idc_probability,
            'prediction_map': prob_map,
            'has_idc': idc_probability > 5  # threshold can be adjusted
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None


def visualize_prediction(image_path, prediction_map, save_path=None):
    """Visualize the prediction by overlaying it on the original image."""
    # Load original image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((48, 48))  # Resize to match prediction map
    image_array = np.array(image)

    # Create heatmap overlay
    heatmap = plt.cm.jet(prediction_map)[:, :, :3]  # Remove alpha channel

    # Blend original image with heatmap
    alpha = 0.4
    blended = (1 - alpha) * image_array / 255.0 + alpha * heatmap
    blended = np.clip(blended, 0, 1)

    # Create figure
    plt.figure(figsize=(12, 4))

    # Plot original image
    plt.subplot(131)
    plt.imshow(image_array)
    plt.title('Original Image')
    plt.axis('off')

    # Plot prediction heatmap
    plt.subplot(132)
    plt.imshow(prediction_map, cmap='jet')
    plt.title('IDC Prediction Map')
    plt.axis('off')

    # Plot blended image
    plt.subplot(133)
    plt.imshow(blended)
    plt.title('Overlay')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_predictions(results, save_dir):
    """
    Visualize predictions with original image, prediction map, and overlay.
    Each row shows results for one image.
    """
    num_images = len(results)
    images_per_plot = 10
    num_plots = (num_images + images_per_plot - 1) // images_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * images_per_plot
        end_idx = min(start_idx + images_per_plot, num_images)
        current_batch = results[start_idx:end_idx]
        num_current_images = len(current_batch)

        # Create figure
        fig = plt.figure(figsize=(15, 3 * num_current_images))

        for idx, result in enumerate(current_batch):
            # Load original image
            image = Image.open(result['image']).convert('RGB')
            image = image.resize((48, 48))
            image_array = np.array(image)

            # Create heatmap
            prediction_map = result['prediction_map']
            heatmap = plt.cm.jet(prediction_map)[:, :, :3]

            # Create overlay
            alpha = 0.4
            blended = (1 - alpha) * image_array / 255.0 + alpha * heatmap
            blended = np.clip(blended, 0, 1)

            # Plot original image
            plt.subplot(num_current_images, 3, idx * 3 + 1)
            plt.imshow(image_array)
            plt.title('Original Image')
            plt.axis('off')

            # Plot prediction heatmap
            plt.subplot(num_current_images, 3, idx * 3 + 2)
            plt.imshow(prediction_map, cmap='jet')
            plt.colorbar(label='IDC Probability')
            plt.title('IDC Prediction Map')
            plt.axis('off')

            # Plot overlay with prediction text
            plt.subplot(num_current_images, 3, idx * 3 + 3)
            plt.imshow(blended)
            prediction_text = f"IDC: {'Yes' if result['has_idc'] else 'No'}\n"
            prediction_text += f"Confidence: {result['probability']:.1f}%"
            plt.title(prediction_text,
                      color='red' if result['has_idc'] else 'green')
            plt.axis('off')

            # Add filename as text
            plt.figtext(0.01, 1 - (idx + 0.5) / num_current_images,
                        os.path.basename(result['image']),
                        fontsize=8)

        # Adjust layout and save
        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f'predictions_batch_{plot_idx + 1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"Saved visualization batch {plot_idx + 1} to {save_path}")


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


def process_and_visualize(image_paths, model, device, transform, save_dir, threshold=5.0):
    """Process images and create detailed visualizations."""
    results = []

    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)

            # Calculate IDC probability
            prob_map = output.squeeze().cpu().numpy()
            idc_probability = float(np.mean(prob_map > 0.5) * 100)

            results.append({
                'file': os.path.basename(image_path),
                'image': image_path,
                'prediction_map': prob_map,
                'probability': idc_probability,
                'has_idc': idc_probability > threshold,
                'confidence_level': create_confidence_description(idc_probability)
            })

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    # Create visualizations
    if results:
        visualize_predictions(results, save_dir)

        # Save detailed results to text file
        results_file = os.path.join(save_dir, 'results.txt')
        with open(results_file, 'w') as f:
            f.write("IDC Detection Results\n")
            f.write("===================\n\n")
            for r in results:
                f.write(f"File: {r['file']}\n")
                f.write(f"IDC Prediction: {
                        'Positive' if r['has_idc'] else 'Negative'}\n")
                f.write(f"Probability: {r['probability']:.2f}%\n")
                f.write(f"Confidence Level: {r['confidence_level']}\n")
                f.write("-------------------\n")

        # Print summary
        print("\nProcessing Complete!")
        print(f"Total images processed: {len(results)}")
        positive_cases = sum(1 for r in results if r['has_idc'])
        print(f"IDC Positive cases: {positive_cases} ({
              positive_cases/len(results)*100:.1f}%)")
        print(f"Results saved to: {results_file}")

    return results
# Update the main function in inference.py to use these functions:


def main():
    parser = argparse.ArgumentParser(description='IDC Detection Inference')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join("./", 'final_model.pth'),
                        help='Path to trained model')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output_dir', type=str,
                        default="./predictions",
                        help='Directory to save predictions')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='IDC probability threshold (%)')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
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

    # Process images and create visualizations
    process_and_visualize(image_paths, model, device, transform,
                          args.output_dir, args.threshold)


if __name__ == '__main__':
    main()
