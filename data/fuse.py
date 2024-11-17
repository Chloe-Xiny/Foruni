import os
import numpy as np
import torch

def load_tensor(file_path):
    """
    Load the tensor from a given file path.
    """
    return torch.load(file_path)

def save_merged_tensor(merged_tensor, save_path):
    """
    Save the merged tensor to the specified path.
    """
    torch.save(merged_tensor, save_path)

def get_view_tensors(video_path, selected_views):
    """
    Load tensors of the selected views for the specified video.
    """
    tensors = []
    for view in selected_views:
        tensor_path = os.path.join(video_path, f"{os.path.basename(video_path)}_Tensor_{view}.pt")
        if os.path.exists(tensor_path):
            tensors.append(load_tensor(tensor_path))
        else:
            print(f"Warning: Tensor for view {view} not found in {tensor_path}")
    return tensors

def merge_tensors(tensors):
    """
    Merge tensors by concatenating them along the feature dimension.
    """
    return torch.cat(tensors, dim=-1)

def main():
    # Original dataset path
    dataset_path = os.path.join(os.sep, "EchoNet-Dynamic", "Dataset")
    
    # Prompt the user to enter the name of the new dataset to save
    new_dataset_name = input("Enter the name for the new dataset: ")
    new_dataset_path = os.path.join(os.sep, "EchoNet-Dynamic", new_dataset_name)
    os.makedirs(new_dataset_path, exist_ok=True)
    
    # Prompt the user to choose views to merge
    print("Available views: 0, 1, 2, 3, 4")
    selected_views = input("Enter the views to merge, separated by commas (e.g., 0,1,3): ")
    selected_views = [int(view.strip()) for view in selected_views.split(",")]

    # Iterate over each video directory and merge selected views
    for video_name in os.listdir(dataset_path):
        video_path = os.path.join(dataset_path, video_name)
        if os.path.isdir(video_path):
            print(f"Processing video: {video_name}")
            
            # Load tensors of selected views
            tensors = get_view_tensors(video_path, selected_views)
            
            if len(tensors) > 0:
                # Merge tensors
                merged_tensor = merge_tensors(tensors)
                
                # Save the merged tensor to the new dataset directory
                save_path = os.path.join(new_dataset_path, f"{video_name}_Merged_Tensor.pt")
                save_merged_tensor(merged_tensor, save_path)
                print(f"Saved merged tensor for {video_name} to {save_path}")
            else:
                print(f"No valid tensors found for video {video_name} with the selected views.")

if __name__ == "__main__":
    main()
