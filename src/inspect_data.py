import pandas as pd

# Load the dataset
try:
    data = pd.read_csv("data/spam.csv", encoding="latin-1")
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Step 1: Drop irrelevant columns
data = data.iloc[:, :2]  # Keep only the first two columns
data.columns = ['label', 'message']  # Rename columns
print("\nAfter dropping irrelevant columns:")
print(data.head())

# Step 2: Encode labels (spam -> 1, ham -> 0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print("\nAfter encoding labels:")
print(data.head())

# Step 3: Check for null values and drop them
print(f"\nNumber of null values before dropping: {data.isnull().sum().sum()}")
data = data.dropna()
print(f"Number of null values after dropping: {data.isnull().sum().sum()}")

# Step 4: Remove duplicates
print(f"\nNumber of duplicates before dropping: {data.duplicated().sum()}")
data = data.drop_duplicates()
print(f"Number of duplicates after dropping: {data.duplicated().sum()}")

# Step 5: Save the cleaned data
data.to_csv("data/cleaned_spam.csv", index=False)
print("\nCleaned data saved successfully!")
