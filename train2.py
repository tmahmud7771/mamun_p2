import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
import csv
from datetime import datetime

# [Previous DoubleConv, UNet, and Dataset classes remain the same]


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=48, patch_size=8, in_channels=3, embed_dim=192):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
        # Add position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
    def forward(self, x):
        B = x.shape[0]
        # Split image into patches and embed
        x = self.projection(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embeddings
        x = x + self.positions
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
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
    def __init__(
        self, 
        image_size=48,
        patch_size=8,
        in_channels=3,
        embed_dim=192,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=1
    ):
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
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Classification head
        x = self.norm(x)
        x = self.head(x[:, 0])  # Use CLS token for classification
        return torch.sigmoid(x)

class IDCDataset(Dataset):
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

            return image, torch.tensor(label, dtype=torch.float32)  # No shape modification here
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            return None


def save_model_summary(model, input_size=(1, 3, 48, 48)):
    """Save model summary to a text file"""
    try:
        # Create directory if it doesn't exist
        os.makedirs('metrics', exist_ok=True)

        # Open file with UTF-8 encoding
        with open(os.path.join('metrics', 'model_summary.txt'), 'w', encoding='utf-8') as f:
            model_summary = summary(model, input_size=input_size, verbose=0)
            print(model_summary, file=f)
    except Exception as e:
        print(f"Warning: Could not save model summary: {str(e)}")
        # Continue execution even if summary saving fails
        pass


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

    # Plot training progress (Loss)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(metrics_df) + 1)  # Create epoch numbers
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

    # Plot Learning Rate
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics_df['learning_rate'], 'b-')
    plt.title('Learning Rate Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'learning_rate_plot.png'))
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


def validate_model(model, val_loader, criterion, device):
    """Validate the model and return metrics"""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    val_progress = tqdm(val_loader, desc='Validating', leave=False)

    with torch.no_grad():
        for inputs, labels in val_progress:
            inputs = inputs.to(device)
            # Reshape labels to match output shape
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Convert predictions to binary
            predictions = (outputs > 0.5).float()
            
            # Adjust predictions and labels for metrics
            predictions = predictions.squeeze()
            labels = labels.squeeze()
            
            all_preds.extend(predictions.cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy().astype(int))

            val_progress.set_postfix({'val_loss': f'{loss.item():.4f}'})

    return val_loss / len(val_loader), np.array(all_preds), np.array(all_labels)
   



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
            # Reshape labels to match output shape
            labels = labels.to(device).unsqueeze(1)  # Change this line

            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            
            # Adjust predictions and labels for metrics
            predictions = predictions.squeeze()
            labels = labels.squeeze()
            
            all_preds.extend(predictions.cpu().numpy().astype(int))
            all_labels.extend(labels.cpu().numpy().astype(int))

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation phase
        val_loss, val_preds, val_labels = validate_model(
            model, val_loader, criterion, device)

        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        epoch_time = time.time() - epoch_start_time

        # Ensure binary format for confusion matrix
        val_preds = val_preds.astype(int)
        val_labels = val_labels.astype(int)

        try:
            cm = confusion_matrix(val_labels, val_preds)
            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = precision_score(val_labels, val_preds, zero_division=0)
            recall = recall_score(val_labels, val_preds, zero_division=0)
            f1 = f1_score(val_labels, val_preds, zero_division=0)
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            continue

        metrics = {
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

    # Convert metrics list to DataFrame and plot
    metrics_df = pd.DataFrame(metrics_list)
    plot_training_metrics(metrics_df)

    # Plot final training metrics
    metrics_df = pd.DataFrame(metrics_list)
    plot_training_metrics(metrics_df)

    return metrics_df


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip(),  # Add data augmentation
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    ])

    # Create dataset
    print("Loading dataset...")
    full_dataset = IDCDataset('dataset', transform=transform)

    # Split dataset into train and validation
    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)

    # Initialize model, criterion, and optimizer
    print("Initializing model...")
    model = VisionTransformer(
        image_size=48,
        patch_size=8,
        in_channels=3,
        embed_dim=192,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=1
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=LEARNING_RATE, 
                           weight_decay=0.05)

    # Train model
    metrics_df = train_model(model, train_loader, val_loader, criterion,
                             optimizer, device, NUM_EPOCHS)

    # Save final model and metrics
    torch.save(model.state_dict(), 'final_model.pth')
    metrics_df.to_csv('final_metrics.csv', index=False)
    print("Training completed. Model and metrics saved.")


if __name__ == '__main__':
    main()
