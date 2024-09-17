from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np

# Load BLIP model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


transform = transforms.Compose([
    transforms.Resize((384, 384)),  # Resize CIFAR-10 images to 384x384
    transforms.ToTensor(),          # Convert the images to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to match CLIP's expected range
])

cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

# Create DataLoader to iterate over the resized CIFAR-10 data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Get a batch of training data
data_iter = iter(trainloader)
images, labels = next(data_iter)

images = [Image.fromarray(((img.permute(1, 2, 0) * 0.5 + 0.5) * 255).clamp(0, 255).byte().numpy()) for img in images]
inputs = processor(images=images, return_tensors="pt")
# Move the images to the correct device
inputs = inputs.to(device)

# Preprocess the images using CLIP's preprocess function (already done with Resize and Normalize)
# images = preprocess(Image.fromarray(images.numpy()))  # This is if you're not using the transform pipeline

# Encode the images using the CLIP model
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=30)
    predicted_labels = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Convert the label index to human-readable label
predicted_label = predicted_labels
ground_truth_label = cifar10_labels[labels[0].item()]

# Display results
print(f"pred:{predicted_label}, y:{ground_truth_label}")
