import pandas as pd

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv('image_classification_results.csv')

# Step 2: Extract the numeric part of the filename (remove '.jpg' and convert to int)
df['numeric_value'] = df['Image'].str.replace('.jpg', '', regex=False).astype(int)

# Step 3: Sort the DataFrame by the numeric part
df_sorted = df.sort_values(by='numeric_value')

# Step 4: Drop the helper column if you no longer need it
df_sorted = df_sorted.drop(columns=['numeric_value'])

# Step 5: Save the sorted DataFrame back to a CSV file or display it
df_sorted.to_csv('sorted_image_classification_results.csv', index=False)

# Optionally, display the sorted DataFrame
print(df_sorted)