import os
import sys
import pickle
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the project root to sys.path (assuming src is in the root directory)
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(project_root)

from src.data_processing.FormalityTransferDataset import FormalityTransferDataset

# paths
test_path = os.path.join(os.getcwd(), 'data/processed/test_dataset_processed.pkl')
sys.path.append(os.path.join(os.getcwd(), 'src/data_processing'))
#print(sys.path)

# Load datasets
with open(test_path, 'rb') as f:
    test : FormalityTransferDataset = pickle.load(f)

# Assuming `test` already contains the structure {'input_ids', 'attention_mask', 'labels'}
#test_dataset = FormalityTransferDataset(test)
sliced_dataset = test.get_slice(0, 100)

# Check the length and structure
print(f"Length of sliced dataset: {len(sliced_dataset)} VS Length of original dataset: {len(test)}")
print(test[0])  # Show a sample