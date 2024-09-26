import pdb

import torch
from PIL import Image
import clip
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from zero_shot_multiclass import CLIPOnlyNetwork, HybridNetwork  # Import your models from zero_shot_multiclass.py

device = "cuda" if torch.cuda.is_available() else "cpu"

categories = [
    "fashion", "food", "travel", "selfie", "group photo",
    "pets", "fitness", "art", "nature", "beauty",
    "products", "quotes", "lifestyle", "events", "technology"
]

# Load the pre-trained model
model_path = "Models/likes_classification_model.pth"
use_hybrid = False  # Set to True if you're using the hybrid model
state_dict = torch.load(model_path, map_location=device)
# Check if the state dict keys are wrapped in 'module.'
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("module."):
        new_key = key[len("module."):]  # Remove 'module.' from the key
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

if use_hybrid:
    model = HybridNetwork(input_size_meta=532, hidden_size=128, output_size=6)
else:
    model = CLIPOnlyNetwork(input_size=532, hidden_size=128, output_size=6)

model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

# Load the CLIP model and preprocessing
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load the original training data to refit the scaler
train_data = pd.read_csv("instagram_data.csv")  # Replace with your actual training data file
scaler = MinMaxScaler()

# Function to preprocess the image and metadata for CLIP-only model
def preprocess_input(image_path, category, no_of_comments, follower_count_at_t):
    # Preprocess image using CLIP
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    # Encode image using CLIP
    with torch.no_grad():
        image_features = clip_model.encode_image(image).cpu().numpy().flatten()

    metadata = pd.DataFrame({
        "no_of_comments": [no_of_comments],
        "follower_count_at_t": [follower_count_at_t],
    })
    metadata[['followers_normalized', 'comments_normalized']] = scaler.fit_transform(
        metadata[['follower_count_at_t', 'no_of_comments']])
    metadata["Image Features"] = [image_features]
    metadata_values = metadata.apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(np.float32)

    category_encoded = np.zeros(len(categories))  # Create a one-hot encoded vector for the category
    category_index = categories.index(category)  # Find the index of the category in the list
    category_encoded[category_index] = 1
    metadata_final = np.hstack((metadata_values, category_encoded.reshape(1, -1)))

    combined_input = np.hstack((metadata_final, image_features.reshape(1,-1))).astype(np.float32)
    print(f"Combined input shape: {combined_input.shape}")

    if use_hybrid:
        return image, torch.tensor(combined_input).to(device)
    return torch.tensor(combined_input).to(device)


# Function to make predictions
def predict(image_path, category, no_of_comments, follower_count_at_t):
    input_data = preprocess_input(image_path, category, no_of_comments, follower_count_at_t)
    with torch.no_grad():
        if use_hybrid:
            outputs = model(input_data[0], input_data[1])  # For hybrid model (CNN + metadata + CLIP features)
        else:
            outputs = model(input_data)  # For CLIP-only model (metadata + CLIP features)

        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item()

# Modify Below to test =======================
image_path = "insta_data/0.jpg"  # Replace with actual image path
no_of_comments = 0
t = 1594174009
follower_count_at_t = 40934474
likes = 154552
category = "travel" # pick one that is closest from the Categories above
# ================================

predicted_label = predict(image_path, category, no_of_comments, follower_count_at_t)
print(f"Predicted Likes Class: {predicted_label}")
print(f"Actual Likes: {likes}")
