import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from config import NUM_CLASSES
from model import LeNet5  # Import your model architecture (adjust according to your file structure)

# Define the transformation to match the input format of LeNet-5
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize image to 32x32 as expected by LeNet-5
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.ToTensor(),  # Convert the image to a tensor
])


def load_model(model_path):
    """
    Load the trained model weights from the given path.
    """
    model = LeNet5(num_classes=NUM_CLASSES)  # Initialize your model
    model.load_state_dict(torch.load(model_path))  # Load the saved model weights
    model.eval()  # Set the model to evaluation mode
    return model


def predict_image(image_path, model):
    """
    Predict the digit in the image using the trained model.
    """
    # Open the image
    image = Image.open(image_path)

    # Apply the transformation to the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)  # Forward pass through the model
        predicted = torch.argmax(torch.softmax(output, dim=1), dim=1)  # Get the predicted class (digit)

    return predicted.item()


def visualize_image(image_path):
    """
    Display the input image.
    """
    image = Image.open(image_path)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    MODELS_FOLDER = os.environ['MODELS_FOLDER']
    SAMPLE_IMAGE_PATH = os.environ['SAMPLE_IMAGE_PATH']

    # Load the trained model
    lenet_model = load_model(MODELS_FOLDER + '/lenet5_model.pth')

    # Predict the digit in the input image
    predicted_class = predict_image(SAMPLE_IMAGE_PATH, lenet_model)

    # Print the prediction
    print(f"Predicted Digit: {predicted_class}")

    # Visualize the image
    visualize_image(SAMPLE_IMAGE_PATH)
