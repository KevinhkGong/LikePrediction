import torch
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from torchvision import transforms

# Load the Vision Transformer model and the feature extractor
model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name, num_labels=6)  # 6 classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap model with DataParallel to allow multi-GPU usage
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)

model.to(device)


# Custom Dataset to load images and labels
class InstagramDataset(Dataset):
    def __init__(self, dataframe, image_dir, feature_extractor, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx]['Image'])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Apply the feature extractor for Vision Transformer
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Flatten to remove the batch dimension
        inputs['pixel_values'] = inputs['pixel_values'].squeeze()

        # Extract the label (make sure 'likes_class' exists in the dataframe)
        label = torch.tensor(self.dataframe.iloc[idx]['likes_class'])

        return inputs['pixel_values'], label


# Preprocessing the images for ViT
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224 for ViT
    transforms.ToTensor(),
])

# Load your metadata
metadata = pd.read_csv("instagram_data.csv")
metadata['Image'] = metadata['image_path'].apply(lambda x: os.path.basename(x))
metadata['likes_class'] = pd.cut(metadata['likes'], bins=[0, 1000, 10000, 50000, 100000, 200000, metadata['likes'].max()],
                                 labels=[0, 1, 2, 3, 4, 5])

# Train/test split
train_data, test_data = train_test_split(metadata, test_size=0.2, random_state=42)

# Create Dataset instances
train_dataset = InstagramDataset(train_data, image_dir="insta_data", feature_extractor=feature_extractor,
                                 transform=transform)
test_dataset = InstagramDataset(test_data, image_dir="insta_data", feature_extractor=feature_extractor,
                                transform=transform)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Optimizer and Loss Function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(pixel_values=inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=inputs).logits
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
