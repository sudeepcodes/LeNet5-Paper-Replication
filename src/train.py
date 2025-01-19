import torch
from torchvision import transforms

from model import LeNet5
from config import BATCH_SIZE, NUM_CLASSES, LEARNING_RATE, NUM_EPOCHS, DEVICE
from data_setup import create_dataloaders
from engine import train
from utils import save_model

# Create Transform
data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Create DataLoaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    transform=data_transform,
    batch_size=BATCH_SIZE)

# Create Model
model = LeNet5(NUM_CLASSES, True)

# Setup loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE)

# Start training the model
result = train(model=model,
               train_dataloader=train_dataloader,
               test_dataloader=test_dataloader,
               loss_fn=loss_fn,
               optimizer=optimizer,
               epochs=NUM_EPOCHS,
               device=DEVICE)

# Save the model with help from utils.py
save_model(model=model,
           target_dir="../models",
           model_name="lenet5_model.pth")
