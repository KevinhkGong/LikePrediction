import pdb

import torch
import clip
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import joblib


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"Using device: {device}")

categories = [
    "fashion", "food", "travel", "selfie", "group photo",
    "pets", "fitness", "art", "nature", "beauty",
    "products", "quotes", "lifestyle", "events", "technology"
]
text_inputs = clip.tokenize(categories).to(device)
image_dir = "insta_data"
results = []



if os.path.exists('exported_data/image_classification_results_no_feature.csv'):
    results_df = pd.read_csv('exported_data/image_classification_results_no_feature.csv')
else:
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

                # Append the result to the results list
                results.append({"Image": image_file, "Predicted Category": top_category})
                print(f"Done with {image_file}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        results_df = pd.DataFrame(results)
        results_df.to_csv("image_classification_results_no_feature.csv", index=False)

print(results_df)



metadata = pd.read_csv("instagram_data.csv")
metadata['Image'] = metadata['image_path'].apply(lambda x: os.path.basename(x))  # Extract just the file name

metadata = pd.merge(metadata, results_df, on="Image")

metadata['followers_normalized'] = (metadata['follower_count_at_t'] - metadata['follower_count_at_t'].min()) / (metadata['follower_count_at_t'].max() - metadata['follower_count_at_t'].min())
metadata['comments_normalized'] = (metadata['no_of_comments'] - metadata['no_of_comments'].min()) / (metadata['no_of_comments'].max() - metadata['no_of_comments'].min())
print("min follower", metadata['follower_count_at_t'].min())
print("max follower", metadata['follower_count_at_t'].max())
print("min comment", metadata['no_of_comments'].min())
print("max comment", metadata['no_of_comments'].max())
metadata = pd.concat([metadata, pd.get_dummies(metadata['Predicted Category'], prefix='category')], axis=1)

print("max", metadata['likes'].max())
metadata['likes_class'] = pd.cut(metadata['likes'], bins= [0, 200000, 400000, metadata['likes'].max()], labels=[0, 1, 2])

X = metadata.drop(columns=['likes', 'likes_class', 'Image', 'Predicted Category', 'image_path', 't'])  # Features
y = metadata['likes_class']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save Model
feature_names = X.columns.tolist()
joblib.dump((clf,feature_names), "Models/random_forest_threeClass.pkl")
print("Model Saved")