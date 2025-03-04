import torch
from torch.utils.data import Dataset, DataLoader


class UCIHAR_Dataset(Dataset):

    def __init__(self, data_path, labels_path):
        """
        Args:
            data_path (str): Path to the tensor containing input features.
            labels_path (str): Path to the tensor containing labels.
        """
        self.data = torch.load(data_path)
        self.labels = torch.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloader(batch_size, train):
    """
    Creates and returns a DataLoader for UCI HAR.

    Args:
        batch_size (int): Number of samples per batch.
        train (bool): Load training set if True, test set if False.
    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    dataset_type = "train" if train else "test"
    dataset = UCIHAR_Dataset(f"uci_har_tensors/X_{dataset_type}.pt",
                             f"uci_har_tensors/y_{dataset_type}.pt")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
