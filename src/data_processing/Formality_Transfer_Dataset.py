import torch
from torch.utils.data.dataset import Dataset

# Define a formality transfer dataset class
class FormalityTransferDataset(Dataset):
    def __init__(self, data_dict):
        self.input_ids = data_dict['input_ids']
        self.attention_mask = data_dict['attention_mask']

    def __len__(self):
        return len(self.input_ids)  # Number of samples

    def __getitem__(self, idx):
        return {
                'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
                'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long)
            }

    # Define a method to return a sliced dataset
    def get_slice(self, start, end):
        sliced_data = {
            'input_ids': self.input_ids[start:end],
            'attention_mask': self.attention_mask[start:end]
        }
        return FormalityTransferDataset(sliced_data)