# Instagram Likes Prediction
**Presentation:** [**Slides**](https://docs.google.com/presentation/d/1ybqkFQ-0CWpl1irJLGifZJj1PY5MvviVbi2z9hf3FJY/edit?usp=sharing)
## Overview
This project explores the prediction of Instagram post likes using a combination of image data and post metadata (e.g., comments, followers). The repository implements and compares various machine learning models to determine which model best predicts the number of likes on Instagram posts.

The models implemented are:
- **Zero-Shot CLIP Neural Network**: Combines image features extracted by CLIP with metadata for predictions.
- **Vision Transformer (ViT)**: A state-of-the-art model designed for image classification, fine-tuned on Instagram images.
- **Random Forest Classifier**: A traditional machine learning algorithm that processes metadata and image features to classify posts into like bins.

## Models Implemented
1. **Zero-Shot CLIP Neural Network**:
   - Uses CLIP for feature extraction from images.
   - Combines CLIP image features with metadata (comments, follower count, etc.).
   - Explores various model configurations (with/without softmax, image features, and category encoding).
   
2. **Vision Transformer (ViT)**:
   - Pre-trained ViT model fine-tuned on Instagram data to predict likes.
   - Performs feature extraction and resizes images to match ViT’s required input size (224x224).
   
3. **Random Forest Classifier**:
   - Trained using metadata (comments, followers) and additional image features.
   - Outperformed the other models in terms of accuracy and speed.

## Features
- **Test Functionality**: A test script is implemented to evaluate the models on new Instagram posts, accepting inputs such as image path, number of comments, timestamp, follower count, and post metadata.
- **Multi-GPU Training**: Models are implemented to support training across multiple GPUs for enhanced performance.
- **Evaluation and Metrics**: Each model’s performance is evaluated using accuracy across various bin settings. The repository also includes performance comparisons between different configurations (e.g., with/without softmax, image features, etc.).

## Dataset
The dataset consists of 3785 Instagram posts, each with:
- **Image data**: Associated image of the post.
- **Metadata**: Including the number of likes, comments, timestamp, and follower count at the time of the post.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/instagram-likes-prediction.git
    cd instagram-likes-prediction
    ```

2. Install the necessary dependencies:
    ```bash
    conda env create -f environment.yml
    ```

3. Download the dataset (metadata and images) and place it in the appropriate directory (`Data/insta_data/`).

## Usage

### Training
To train any of the models, run the corresponding script:
- **Zero-Shot CLIP Model**:
    ```bash
    python zero_shot.py
    ```
- **Vision Transformer Model**:
    ```bash
    python vit_model.py
    ```
- **Random Forest Model**:
    ```bash
    python random_forest.py
    ```

### Testing
You can test the models on new Instagram posts using the provided test script:
- **Test on new data**:
    ```bash
    python test.py 
    ```
    The test script will output the predicted likes class based on the input image and metadata.
    Fill in the information regarding the testing data and run the script

## Results
- The **Random Forest Classifier** achieved the best performance with an accuracy of **~0.82** across multiple bin settings.
- **Zero-Shot CLIP** and **Vision Transformer** models showed comparable results, but the random forest model excelled in handling metadata effectively.

## Limitations and Future Work
- **Data Skewness**: The dataset is highly skewed, favoring lower like counts, which may introduce biases in model predictions.
- **Scalability**: Random Forest performed well in this setting but may become computationally expensive and less scalable as the feature space grows.
- Future improvements could involve augmenting the data, addressing skewness, or trying advanced models like ensemble methods of ViT and Random Forest.
