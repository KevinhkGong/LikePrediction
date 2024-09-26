import pandas as pd

df = pd.read_csv('exported_data/image_classification_results.csv')
df['numeric_value'] = df['Image'].str.replace('.jpg', '', regex=False).astype(int)
df_sorted = df.sort_values(by='numeric_value')
df_sorted = df_sorted.drop(columns=['numeric_value'])
df_sorted.to_csv('sorted_image_classification_results.csv', index=False)

print(df_sorted)