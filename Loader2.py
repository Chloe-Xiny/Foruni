import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

class FusedTensorDataset(Dataset):
    def __init__(self, \data\DataSet):
        
        self.root_dir = root_dir
        self.tensor_files = self._collect_tensor_files()

    def _collect_tensor_files(self):
        
        tensor_files = []
        for tensor_file in os.listdir(self.root_dir):
            if tensor_file.endswith('_FusedTensor.npy'):
                tensor_files.append(os.path.join(self.root_dir, tensor_file))
        return tensor_files

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = self.tensor_files[idx]
        fused_tensor = np.load(tensor_path)
        fused_tensor = torch.tensor(fused_tensor, dtype=torch.float32)
        return fused_tensor


def get_fused_dataloader(root_dir, batch_size=16, shuffle=True, num_workers=4):
   
    dataset = FusedTensorDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fused Tensor DataLoader')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the fused tensor dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    args = parser.parse_args()

    # Create DataLoader for the fused tensor dataset
    dataloader = get_fused_dataloader(root_dir=args.root_dir, batch_size=args.batch_size,
                                      shuffle=args.shuffle, num_workers=args.num_workers)

    # Example usage: iterate through the dataloader
    for batch_idx, data in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: {data.shape}")
