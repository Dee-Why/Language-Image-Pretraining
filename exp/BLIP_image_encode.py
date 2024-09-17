from transformers import BlipProcessor, BlipVisionModel
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np

# Load the BLIP model and processor
model = BlipVisionModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the CIFAR-10 label names
cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Transform CIFAR-10 images to the appropriate size and format
transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Adjust if needed
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# Extract features for a batch
data_iter = iter(dataloader)
images, labels = next(data_iter)

# Convert image to PIL format and process
images_pil = [transforms.ToPILImage()(img) for img in images]
inputs = processor(images=images_pil, return_tensors="pt").to(device)

# Generate image features
with torch.no_grad():
    outputs = model(**inputs)
    image_features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Example pooling

print("Extracted image features:", image_features)

