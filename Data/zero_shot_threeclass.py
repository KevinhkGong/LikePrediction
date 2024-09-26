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
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(device)

categories = [
    "fashion", "food", "travel", "selfie", "group photo",
    "pets", "fitness", "art", "nature", "beauty",
    "products", "quotes", "lifestyle", "events", "technology"
]

text_inputs = clip.tokenize(categories).to(device)
image_dir = "insta_data"
results = []
image_features_list = []  # To store the CLIP image features

# Check if results are already saved
if os.path.exists('exported_data/image_classification_results.csv'):
    results_df = pd.read_csv('exported_data/image_classification_results.csv')
    # Convert the 'Image Features' column back to NumPy arrays
    results_df['Image Features'] = results_df['Image Features'].apply(lambda x: np.fromstring(x, sep=','))
    image_features_list = np.stack(results_df['Image Features'].values)
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
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text_inputs)

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
                image_features_list.append(image_features_flat)
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
pdb.set_trace()
# metadata = pd.concat([metadata, pd.get_dummies(metadata['Predicted Category'], prefix='category')], axis=1)

# Add image features to metadata
metadata_features = metadata.drop(columns=['likes', 'Image', 'Predicted Category', 'image_path', 't'])
# metadata_features = metadata.drop(columns=['likes', 'Image', 'Predicted Category', 'image_path', 't', 'Image Features'])
metadata_values = metadata_features.apply(pd.to_numeric, errors='coerce').fillna(0).values  # Convert metadata to numpy

# Concatenate image features with metadata features
X = np.hstack((metadata_values, image_features_list)).astype(float)  # Concatenate along the feature axis
# X = np.array(metadata_values).astype(float)  # Concatenate along the feature axis
pdb.set_trace()
# Set up target labels
metadata['likes_class'] = pd.cut(metadata['likes'], bins=[0, 200000, 400000, metadata['likes'].max()], labels=[0, 1, 2])
y = metadata['likes_class'].astype(int).values  # Target labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create a PyTorch dataset and dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# Initialize the neural network, loss function, and optimizer
input_size = X_train.shape[1]  # Number of features (metadata + image features)
hidden_size = 128
output_size = 3  # 3 classes (low, medium, high)

model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f"Accuracy on the test set: {accuracy:.4f}")

torch.save(model.state_dict(), "Models/likes_classification_model_threeClass.pth")
