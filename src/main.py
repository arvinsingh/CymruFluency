import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from src.train import train_model


def main(
        dataset,
        labels,
        model_constructor,
        model_kwargs,
        save_path,
        num_epochs=50,
        batch_size=16,
        lr=1e-3,
):
    """
    General main function for training any model with separate stratified train and validation sets.
    
    Parameters:
        model_constructor: A callable that returns a model instance (e.g., a class).
        model_kwargs: A dict of keyword arguments to pass to the model constructor.
        save_path: File path where the trained model will be saved.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training and validation.
    """
    np.random.seed(42)
    
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, valid_idx = next(sss.split(indices, labels))
    
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_constructor(**model_kwargs)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
