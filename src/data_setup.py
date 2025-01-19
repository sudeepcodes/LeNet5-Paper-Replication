from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from config import NUM_WORKERS
from fetch_data import download_data


def create_dataloaders(
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    download_data(data_dir='data')
    train_data = datasets.MNIST(root='data',
                                train=True,
                                download=False,
                                transform=transform)

    test_data = datasets.MNIST(root='data',
                               train=False,
                               download=False,
                               transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,  # don't need to shuffle test data
    )

    return train_dataloader, test_dataloader, class_names
