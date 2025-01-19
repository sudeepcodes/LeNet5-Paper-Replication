import os
from torchvision import datasets
from torchvision import transforms


def download_data(data_dir='data'):
    """
    This function checks if the MNIST dataset is already downloaded in the specified directory.
    If not, it will download the dataset.
    """
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Check if the MNIST dataset is already downloaded
    if os.path.exists(os.path.join(data_dir, 'MNIST')):
        print("MNIST dataset already downloaded.")
    else:
        print("Downloading MNIST dataset...")
        # Define the transformation (resize and convert to tensor)
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        # Download the dataset
        datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        print("MNIST dataset downloaded.")

