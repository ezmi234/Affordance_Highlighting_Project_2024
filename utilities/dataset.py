import pickle
import numpy as np
import torch

# ================== AffordanceNet Dataset HELPER FUNCTIONS =============================

def load_dataset(path):
    """
      Load the affordance dataset from a pickle file.

      Returns a list of dicts with keys:
      - 'shape_id'
      - 'semantic class'
      - 'affordance'
      - 'data_info'
    """
    dataset = []
    with open(path, 'rb') as f:
        train_data = pickle.load(f)
        print("Loaded train_data")
        for index,info in enumerate(train_data):
            temp_info = {}
            temp_info["shape_id"] = info["shape_id"]
            temp_info["semantic class"] = info["semantic class"]
            temp_info["affordance"] = info["affordance"]
            temp_info["data_info"] = info["full_shape"]
            dataset.append(temp_info)
    return dataset

def get_coordinates(sample, device='cpu'):
    """
    Returns the point cloud coordinates from a sample as a torch.Tensor on the specified device.

    Args:
        sample: one entry from the dataset
        device: 'cpu' or 'cuda'

    Returns:
        coords: torch.Tensor of shape [N, 3]
    """
    return torch.tensor(sample["data_info"]["coordinate"], dtype=torch.float32).to(device)

def get_affordance_classes(sample):
    """
    Returns the list of affordance class names available for the given sample.
    """
    return sample["affordance"]

def is_affordance_present(sample, affordance_class):
    """
    Given a sample and an affordance class string (e.g., 'grasp'),
    returns True if there is at least one positive label for that affordance,
    otherwise False.
    """
    label = np.array(sample["data_info"]["label"][affordance_class])
    return np.any(label > 0)

def get_affordance_label(sample, affordance_class, device='cpu'):
    """
    Returns the binary label mask for a specific affordance class as a torch tensor.

    Args:
        sample: dataset sample
        affordance_class: string key of the affordance label
        device: 'cpu' or 'cuda'

    Returns:
        labels: torch.Tensor of shape [N]
    """
    label = sample["data_info"]["label"][affordance_class]
    return torch.tensor(label, dtype=torch.float32).squeeze().to(device)