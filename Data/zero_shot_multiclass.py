import pdb

import torch
import clip
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Define the CLIP-only neural network model
class CLIPOnlyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CLIPOnlyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, meta):
        x = self.fc1(meta)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Define the hybrid neural network model (with raw image + CLIP features)
class HybridNetwork(nn.Module):
    def __init__(self, input_size_meta, hidden_size, output_size):
        super(HybridNetwork, self).__init__()

        # Branch to handle image input (convolutional layers)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layers for metadata + CLIP features
        self.fc1 = nn.Linear(input_size_meta + 64 * 28 * 28, hidden_size)  # Adjust size based on CNN output
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, img, meta):
        # CNN branch for image input
        cnn_output = self.cnn_layers(img)

        # Concatenate CNN output with metadata and CLIP features
        x = torch.cat((cnn_output, meta), dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class ImageMetadataDataset(Dataset):
    def __init__(self, image_paths, metadata, transform=None):
        self.image_paths = image_paths
        self.metadata = metadata
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Load image
        if self.transform:
            image = self.transform(image)

        meta = torch.tensor(self.metadata[idx], dtype=torch.float32)  # Load metadata/CLIP features
        return image, meta

def main():
    # Detect if multiple GPUs are available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Using device: {device}, Number of GPUs: {num_gpus}")

    categories = [
        "fashion", "food", "travel", "selfie", "group photo",
        "pets", "fitness", "art", "nature", "beauty",
        "products", "quotes", "lifestyle", "events", "technology"
    ]

    text_inputs = clip.tokenize(categories).to(device)
    image_dir = "insta_data"
    results = []
    image_features_list = []  # To store the CLIP image features

    # Mode selection: Set to True for hybrid mode, False for CLIP-only mode
    use_hybrid = False
    num_epochs = 50

    # Check if results are already saved
    if os.path.exists('exported_data/image_classification_results.csv'):
        results_df = pd.read_csv('exported_data/image_classification_results.csv')

        # Convert the 'Image Features' column back to NumPy arrays
        results_df['Image Features'] = results_df['Image Features'].apply(lambda x: np.fromstring(x, sep=','))
        image_features_list = np.stack(results_df['Image Features'].values)  # Stack them back into an array

    else:
        # Process images with CLIP and extract features
        for image_file in os.listdir(image_dir):
            if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Check for image files
                image_path = os.path.join(image_dir, image_file)
                try:
                    # Preprocess the image
                    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                    # Perform the classification using CLIP
                    with torch.no_grad():
                        # Encode the image and text categories
                        image_features = clip_model.encode_image(image)
                        text_features = clip_model.encode_text(text_inputs)
                        # Normalize the features
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        text_features /= text_features.norm(dim=-1, keepdim=True)

                        # Compute cosine similarity between image and text categories
                        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                    # Extract the predicted category with the highest probability
                    probs = similarities.cpu().numpy()[0]
                    top_category = categories[probs.argmax()]

                    # Store the extracted image features for this image
                    image_features_flat = image_features.cpu().numpy().flatten()  # Flatten the image features
                    image_features_list.append(image_features_flat)  # Store for later use

                    # Convert image features to a list of strings for CSV storage
                    image_features_str = ','.join(map(str, image_features_flat))

                    # Append the result to the results list
                    results.append({
                        "Image": image_file,
                        "Predicted Category": top_category,
                        "Image Features": image_features_str  # Store flattened image features
                    })
                    print("done with", image_file)

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

        # Save both predicted categories and image features in the same CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv("image_classification_results.csv", index=False)

    print(results_df)

    # Load metadata and merge with the results_df
    metadata = pd.read_csv("instagram_data.csv")
    metadata['Image'] = metadata['image_path'].apply(lambda x: os.path.basename(x))
    metadata = pd.merge(metadata, results_df, on="Image")
    print(metadata)

    # Normalize followers and comments
    scaler = MinMaxScaler()
    metadata[['followers_normalized', 'comments_normalized']] = scaler.fit_transform(
        metadata[['follower_count_at_t', 'no_of_comments']])

    # One-hot encode the predicted categories
    metadata = pd.concat([metadata, pd.get_dummies(metadata['Predicted Category'], prefix='category')], axis=1)

    # Add image features to metadata
    metadata_features = metadata.drop(columns=['likes', 'Image', 'Predicted Category', 'image_path', 't'])
    metadata_values = metadata_features.apply(pd.to_numeric, errors='coerce').fillna(0).values  # Convert metadata to numpy

    # Concatenate image features with metadata features
    metadata_values = metadata_values.astype(np.float32)
    X_metadata = np.hstack((metadata_values, image_features_list))  # Concatenate metadata with CLIP features
    pdb.set_trace()
    # Set up target labels
    metadata['likes_class'] = pd.cut(metadata['likes'], bins=[0, 1000, 10000, 50000, 100000, 200000, metadata['likes'].max()],
                                     labels=[0, 1, 2, 3, 4, 5])
    y = metadata['likes_class'].astype(int).values  # Target labels

    X_train_meta, X_test_meta, y_train, y_test = train_test_split(X_metadata, y, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)


    # Create dataset and dataloader
    transform = preprocess  # Use CLIP's preprocessing
    image_paths = metadata['image_path'].values  # List of image paths
    dataset = ImageMetadataDataset(image_paths, X_metadata, transform=transform)

    # Create a DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


    # Choose model based on the mode
    if use_hybrid:
        # Hybrid model (raw image + CLIP + metadata)
        input_size_meta = X_train_meta.shape[1]  # Metadata + CLIP feature size
        hidden_size = 128
        output_size = 6  # 6 classes (based on likes bins)
        model = HybridNetwork(input_size_meta, hidden_size, output_size)
    else:
        # CLIP-only model (metadata + CLIP features)
        input_size = X_train_meta.shape[1]  # Metadata + CLIP feature size
        hidden_size = 128
        output_size = 6  # 6 classes (based on likes bins)
        model = CLIPOnlyNetwork(input_size, hidden_size, output_size)

    # Parallelize the model if more than one GPU is available
    if num_gpus > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("Start Training")
        for img_batch, meta_batch in train_loader:  #
            img_batch = img_batch.to(device)
            meta_batch = meta_batch.to(device)
            labels = y_train_tensor[:len(meta_batch)].to(device)  # Ensure labels match the batch size

            optimizer.zero_grad()
            if use_hybrid:
                outputs = model(img_batch, meta_batch)  # Hybrid model
            else:
                outputs = model(meta_batch)  # CLIP-only model

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img_batch, meta_batch in train_loader:
            img_batch = img_batch.to(device)
            meta_batch = meta_batch.to(device)

            # Get predictions from the model
            if use_hybrid:
                outputs = model(img_batch, meta_batch)  # Hybrid model
            else:
                outputs = model(meta_batch)  # CLIP-only model

            _, predicted = torch.max(outputs, 1)  # Get the predicted class

            # Get the ground truth labels for this batch
            labels = y_test_tensor[:len(predicted)].to(device)  # Ensure the labels match the batch size

            # Update the correct predictions count
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy:.4f}")

    torch.save(model.state_dict(), "Models/likes_classification_model.pth")

if __name__ == "__main__":
    main()