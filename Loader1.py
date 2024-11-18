import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse

class MultiModalDataset(Dataset):
    def __init__(self, \data\Dataset, selected_views, dataset_name):
        """
        Args:
            root_dir (string): Directory with all the videos and extracted tensors.
            selected_views (list): List of selected view indices for fusion (e.g., [0, 1, 3, 4]).
            dataset_name (string): Name of the new dataset.
        """
        self.root_dir = root_dir
        self.selected_views = selected_views
        self.dataset_name = dataset_name
        self.tensor_files = self._collect_tensor_files()

    def _collect_tensor_files(self):
        """Collects paths to all the tensor files based on selected views."""
        tensor_files = []
        for video_name in os.listdir(self.root_dir):
            video_dir = os.path.join(self.root_dir, video_name)
            if os.path.isdir(video_dir):
                tensor_paths = []
                for view in self.selected_views:
                    tensor_file = os.path.join(video_dir, f"{video_name}_Tensor_{view}.npy")
                    if os.path.exists(tensor_file):
                        tensor_paths.append(tensor_file)
                    else:
                        raise FileNotFoundError(f"Tensor file not found: {tensor_file}")
                tensor_files.append(tensor_paths)
        return tensor_files

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_paths = self.tensor_files[idx]
        # Load and concatenate tensors from selected views
        tensors = [np.load(tensor_path) for tensor_path in tensor_paths]
        fused_tensor = np.concatenate(tensors, axis=-1)  # Concatenate along feature dimension
        fused_tensor = torch.tensor(fused_tensor, dtype=torch.float32)
        return fused_tensor

def get_dataloader(root_dir, selected_views, dataset_name, batch_size=16, shuffle=True, num_workers=4):

    
    




    dataset = MultiModalDataset(root_dir=root_dir, selected_views=selected_views, dataset_name=dataset_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-Modal DataLoader')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset (e.g., /EchoNet-Dynamic/Dataset)')
    parser.add_argument('--views', type=int, nargs='+', required=True, help='Selected view indices for fusion (e.g., 0 1 3 4)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the new fused dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')

    args = parser.parse_args()

    # Create DataLoader for the new fused dataset
    dataloader = get_dataloader(root_dir=args.root_dir, selected_views=args.views, dataset_name=args.dataset_name,
                                batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    # Example usage: iterate through the dataloader
    for batch_idx, data in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: {data.shape}")
