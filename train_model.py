"""
Train the Skin Cancer Detection Model
This script trains an EfficientNet-B0 model on the skin cancer dataset
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import efficientnet_b0
from tqdm import tqdm

# Configuration
DATA_DIR = "Dataset"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
MODEL_SAVE_PATH = "skin_model.pth"

def main():
    print("=" * 60)
    print("Skin Cancer Detection - Model Training")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Dataset directory '{DATA_DIR}' not found!")
        print("Please ensure the Dataset folder with AK, BCC, and SK subfolders exists.")
        return
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✓ Using device: {device}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print(f"\n📂 Loading dataset from '{DATA_DIR}'...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"✓ Total images: {len(dataset)}")
    print(f"✓ Classes: {dataset.classes}")
    
    # Calculate class distribution
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\n📊 Class Distribution:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} images")
    
    # Split dataset
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = int(VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    print(f"\n🔀 Splitting dataset:")
    print(f"   Train: {train_size} images ({TRAIN_SPLIT*100:.0f}%)")
    print(f"   Val:   {val_size} images ({VAL_SPLIT*100:.0f}%)")
    print(f"   Test:  {test_size} images ({(1-TRAIN_SPLIT-VAL_SPLIT)*100:.0f}%)")
    
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Build model
    print(f"\n🏗️  Building EfficientNet-B0 model...")
    model = efficientnet_b0(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(1280, 3)  # 3 classes: AK, BCC, SK
    model = model.to(device)
    print("✓ Model loaded with ImageNet pretrained weights")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                val_pbar.set_postfix({
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")
        
        print("-" * 60)
    
    # Test evaluation
    print("\n🧪 Evaluating on test set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = test_correct / test_total
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"✓ Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"✓ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
    print("\nYou can now run the Streamlit app using:")
    print("  ./run.sh")
    print("=" * 60)

if __name__ == "__main__":
    main()
