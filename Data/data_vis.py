import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'instagram_data.csv'
data = pd.read_csv(file_path)

# Check the 'likes' column distribution
likes = data['likes'].astype(int)
print(likes)
print(likes.max())

# Plotting the distribution of the 'likes' column
plt.figure(figsize=(10, 6))
plt.hist(likes, bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Likes')
plt.xlabel('Number of Likes')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()