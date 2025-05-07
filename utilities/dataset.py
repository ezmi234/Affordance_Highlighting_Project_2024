import pickle
import numpy as np
import torch
import random

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

def split_dataset(dataset, val_ratio=0.01, seed=42):
    """
    Splits a dataset into validation and test sets using a fixed random seed.

    Args:
        dataset (list): the full dataset (e.g., list of dicts)
        val_ratio (float): fraction to use for val and for test (total = 2 * val_ratio)
        seed (int): random seed for deterministic shuffling

    Returns:
        (val_set, test_set): two subsets of the dataset
    """

    dataset_copy = dataset.copy()
    random.seed(seed)
    random.shuffle(dataset_copy)

    n_val = int(val_ratio * len(dataset_copy))
    val_set = dataset_copy[:n_val]
    test_set = dataset_copy[n_val:2 * n_val]

    return val_set, test_set

def split_dataset_by_class_and_affordance(dataset, object_class, affordance, val_size=5, test_size=5, seed=42):
    """
    Splits a dataset into validation and test sets for a specific object class and affordance.

    Args:
        dataset (list): the full dataset (e.g., list of dicts)
        object_class (str): the class of the object (e.g., "knife")
        affordance (str): the affordance to filter (e.g., "grasp")
        val_size (int): number of samples to use for validation and test (total = 2 * val_size)
        seed (int): random seed for deterministic shuffling

    Returns:
        (val_set, test_set): two subsets of the dataset
    """
    filtered_dataset = [item for item in dataset if item['semantic class'] == object_class
                        and is_affordance_present(item, affordance)]

    random.seed(seed)
    random.shuffle(filtered_dataset)

    val_set = filtered_dataset[:val_size]
    test_set = filtered_dataset[val_size:val_size + test_size]

    return val_set, test_set