import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import clip  # Need to install: pip install git+https://github.com/openai/CLIP.git
from PIL import Image
import os
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import csv
from torchinfo import summary
import pandas as pd
import os


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(clip_model.visual.output_dim, 2)

        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Initialize classifier in the same dtype as CLIP
        self.classifier = self.classifier.half()

    def forward(self, image):
        # Ensure input is in float16
        if image.dtype != torch.float16:
            image = image.half()

        with torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(image)
            output = self.classifier(image_features)

        # Convert back to float32 for loss computation
        return output.float()


class IDCDatasetCLIP(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Load non-IDC images (class 0)
        class0_path = os.path.join(root_dir, '0')
        for img_name in os.listdir(class0_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(class0_path, img_name))
                self.labels.append(0)

        # Load IDC images (class 1)
        class1_path = os.path.join(root_dir, '1')
        for img_name in os.listdir(class1_path):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.images.append(os.path.join(class1_path, img_name))
                self.labels.append(1)

        print(f"Loaded {len(self.labels)} images")
        print(f"Class 0 (non-IDC): {self.labels.count(0)} images")
        print(f"Class 1 (IDC): {self.labels.count(1)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image_path = self.images[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None


def save_model_summary(model, input_size=(1, 3, 224, 224)):
    """Save model summary to a text file"""
    try:
        os.makedirs('metrics', exist_ok=True)
        with open(os.path.join('metrics', 'model_summary.txt'), 'w', encoding='utf-8') as f:
            model_summary = summary(model, input_size=input_size, verbose=0)
            print(model_summary, file=f)
    except Exception as e:
        print(f"Warning: Could not save model summary: {str(e)}")


def plot_confusion_matrix(cm, classes, epoch, save_dir='plots'):
    """Plot and save confusion matrix"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_epoch_{epoch}.png'))
    plt.close()


def plot_training_metrics(metrics_df, save_dir='plots'):
    """Plot training metrics"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert metrics list to DataFrame if it's not already
    if not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame(metrics_df)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(metrics_df) + 1)
    plt.plot(epochs, metrics_df['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, metrics_df['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    # Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_df['f1_score'], 'g-')
    plt.title('F1 Score Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'f1_score_plot.png'))
    plt.close()

    # Plot Accuracy, Precision, and Recall
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_df['accuracy'], 'b-', label='Accuracy')
    plt.plot(epochs, metrics_df['precision'], 'r-', label='Precision')
    plt.plot(epochs, metrics_df['recall'], 'g-', label='Recall')
    plt.title('Model Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'metrics_plot.png'))
    plt.close()

    # Save numerical metrics to CSV
    metrics_df.to_csv(os.path.join(
        save_dir, 'training_metrics.csv'), index=False)


def initialize_metrics_csv():
    """Initialize CSV file for tracking metrics"""
    os.makedirs('metrics', exist_ok=True)
    filename = os.path.join('metrics',
                            f'training_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    headers = ['epoch', 'train_loss', 'val_loss', 'accuracy', 'precision',
               'recall', 'f1_score', 'learning_rate', 'epoch_time',
               'true_positives', 'false_positives', 'true_negatives',
               'false_negatives']

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    return filename


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    # Initialize CSV logging
    metrics_filename = initialize_metrics_csv()
    save_model_summary(model)

    best_val_loss = float('inf')
    metrics_list = []

    print("\n=== Starting Training ===")
    print(f"Training on device: {device}")
    print(f"Metrics will be saved to: {metrics_filename}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        epoch_start_time = time.time()

        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating', leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)
                val_preds.extend(predictions.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start_time

        try:
            cm = confusion_matrix(val_labels, val_preds)
            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = precision_score(val_labels, val_preds)
            recall = recall_score(val_labels, val_preds)
            f1 = f1_score(val_labels, val_preds)

            # Save metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'epoch_time': epoch_time,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
            metrics_list.append(metrics)

            # Save to CSV
            with open(metrics_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([metrics[key] for key in metrics.keys()])

            # Plot confusion matrix
            plot_confusion_matrix(cm, ['Non-IDC', 'IDC'], epoch + 1)

            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Time: {epoch_time:.2f} seconds')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'metrics': metrics
                }
                torch.save(checkpoint, 'best_model_clip.pth')
                print("Saved new best model")

        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            continue

    # Plot final training metrics
    metrics_df = pd.DataFrame(metrics_list)
    plot_training_metrics(metrics_df)

    return metrics_df


def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('metrics3', exist_ok=True)
    os.makedirs('plots3', exist_ok=True)
    # Set random seed
    torch.manual_seed(42)

    # Parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-2
    VAL_SPLIT = 0.2

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, _ = clip.load('ViT-B/32', device=device, jit=False)

    print("Initializing model...")
    model = CLIPClassifier(clip_model).to(device)

    # Create custom transforms based on CLIP's requirements
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    # Create dataset
    print("Loading dataset...")
    full_dataset = IDCDatasetCLIP('dataset', transform=transform)

    # Split dataset
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    # Initialize model
    model = CLIPClassifier(clip_model).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # Train model
    metrics_list = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, device, NUM_EPOCHS,
        scaler=scaler
    )

    # Save final model
    torch.save(model.state_dict(), 'final_model_clip.pth')
    print("Training completed. Model saved.")


if __name__ == '__main__':
    main()
