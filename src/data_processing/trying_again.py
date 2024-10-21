import pandas as pd
import os


path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
path = os.path.abspath(os.path.join(path, os.pardir))
path = os.path.join(path, 'data/raw/GYAFC_Corpus')


# Step 1: Read lines from the raw file (without using pandas)
formal_file_path = path + '/Entertainment_Music/tune/formal'
with open(formal_file_path, 'r', encoding='utf-8') as f:
    formal_lines = [line.strip() for line in f.readlines()]  # Strip newline characters

# Step 2: Load the same file into a pandas DataFrame
formal_train_df = pd.read_table(formal_file_path, names=['label'], header=None)

# Step 3: Compare lengths (for debugging)
print(f"Raw file length: {len(formal_lines)}")
print(f"DataFrame length: {len(formal_train_df)}")

# Step 4: Find the first mismatch between raw data and DataFrame
for i, raw_line in enumerate(formal_lines):
    if i >= len(formal_train_df):  # If DataFrame has fewer rows than the file
        print(f"First mismatch at row {i}: Row in raw file missing in DataFrame: {raw_line}")
        break
    elif pd.isna(formal_train_df.iloc[i, 0]):  # Check if DataFrame row is NaN
        print(f"First mismatch at row {i}: Row is NaN in DataFrame but exists in raw file: {raw_line}")
        break
    elif raw_line != str(formal_train_df.iloc[i, 0]).strip():  # Compare row contents
        print(f"First mismatch at row {i}:")
        print(f"Raw file: {raw_line}")
        print(f"DataFrame: {formal_train_df.iloc[i, 0]}")
        break
else:
    print("No mismatches found. Raw file and DataFrame are aligned.")
