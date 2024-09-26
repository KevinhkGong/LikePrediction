import pdb

import torch
import clip
from PIL import Image
import os
import joblib
import pandas as pd
import numpy as np


def predict_instagram_post(image_path, likes, no_of_comments, t, follower_count_at_t, use_multi):
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        # Perform classification with CLIP
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute cosine similarity
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarities.cpu().numpy()[0]
        top_category = categories[probs.argmax()]

        print(f"Predicted Category: {top_category}")

    except Exception as e:
        print(f"Error processing image: {e}")
        return None

    follower_count_min = 187
    follower_count_max = 40934474
    no_of_comments_min = 0
    no_of_comments_max = 733973

    followers_normalized = (follower_count_at_t - follower_count_min) / (follower_count_max - follower_count_min)
    comments_normalized = (no_of_comments - no_of_comments_min) / (no_of_comments_max - no_of_comments_min)


    feature_vector = {
        'follower_count_at_t':[follower_count_at_t],
        'no_of_comments': [no_of_comments],
        'followers_normalized': [followers_normalized],
        'comments_normalized': [comments_normalized],
    }

    # Create dummy variables for categories
    for category in categories:
        feature_vector[f'category_{category}'] = [1 if category == top_category else 0]

    # Convert the feature vector to a DataFrame
    X_new = pd.DataFrame(feature_vector)
    # Reorder
    X_new = X_new.reindex(columns=feature_names, fill_value=0)

    predicted_class = clf.predict(X_new)[0]

    if use_multi:
        # Define six-class labels
        class_mapping = {
            0: "0-1000 likes",
            1: "1K-10K likes",
            2: "10K-50K likes",
            3: "50K-100K likes",
            4: "100K-200K likes",
            5: "200000+ likes"
        }
    else:
        # Define three-class labels
        class_mapping = {
            0: "0-200K likes",
            1: "200K-400K likes",
            2: "400K+ likes"
        }

        # Get the human-readable class label
    predicted_label = class_mapping[predicted_class]
    print(f"Predicted Likes Class: {predicted_label}")


# Modify to use multiclass or not=======================
use_multi = True
# =========================================

# Load the trained RandomForest model
if use_multi:
    model_filename = 'Models/random_forest_multiclass.pkl'
else:
    model_filename = 'Models/random_forest_threeClass.pkl'
clf, feature_names = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Load the CLIP model and preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}")

# Define categories (same as used in training)
categories = [
    "fashion", "food", "travel", "selfie", "group photo",
    "pets", "fitness", "art", "nature", "beauty",
    "products", "quotes", "lifestyle", "events", "technology"
]
text_inputs = clip.tokenize(categories).to(device)

# Modify Below to test =======================
predict_instagram_post(
    image_path= "insta_data/0.jpg",
    likes=154552,
    no_of_comments=0,
    t=1594174009,
    follower_count_at_t=40934474,
    use_multi=use_multi
)
# ============================================