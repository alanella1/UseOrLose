"""
1 WALKING
2 WALKING_UPSTAIRS
3 WALKING_DOWNSTAIRS
4 SITTING
5 STANDING
6 LAYING
"""
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve
import zipfile

CWD = os.getcwd()


def load_text_data(split='train'):
    data_path = os.path.join(CWD, 'DATA', split, 'X_' + split + '.txt')
    label_path = os.path.join(CWD, 'DATA', split, 'Y_' + split + '.txt')

    data = np.loadtxt(data_path)
    labels = np.loadtxt(label_path).astype(int) - 1

    return data, labels


# Load train and test data
X_train, y_train = load_text_data("train")
X_test, y_test = load_text_data("test")

# Standardize based on train statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform train
X_test = scaler.transform(X_test)  # Transform test using the same mean/std

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create directory for saving tensors
save_dir = "uci_har_tensors"
os.makedirs(save_dir, exist_ok=True)

# Save tensors
torch.save(X_train_tensor, f"{save_dir}/X_train.pt")
torch.save(y_train_tensor, f"{save_dir}/y_train.pt")
torch.save(X_test_tensor, f"{save_dir}/X_test.pt")
torch.save(y_test_tensor, f"{save_dir}/y_test.pt")

print("Preprocessing complete. Tensors saved to disk.")
