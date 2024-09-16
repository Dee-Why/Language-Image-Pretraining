import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

# Define a transformation to resize the CIFAR-10 images to 224x224
# We will use the CLIP's preprocess to ensure the correct transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize CIFAR-10 images to 224x224
    transforms.ToTensor(),          # Convert the images to a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to match CLIP's expected range
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

# Create DataLoader to iterate over the resized CIFAR-10 data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Get a batch of training data
data_iter = iter(trainloader)
images, labels = next(data_iter)

# Move the images to the correct device
images = images.to(device)

# Preprocess the images using CLIP's preprocess function (already done with Resize and Normalize)
# images = preprocess(Image.fromarray(images.numpy()))  # This is if you're not using the transform pipeline

# Encode the images using the CLIP model
with torch.no_grad():
    image_features = model.encode_image(images)

# The resulting image_features are now ready for downstream tasks like classification
print(f"Image features shape: {image_features.shape}")  # Should print something like torch.Size([64, 512])

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')  # Load in bytes format as required
    return dict_data

def load_cifar10_meta(data_dir):
    meta_file = os.path.join(data_dir, 'batches.meta')
    meta_data = unpickle(meta_file)
    label_names = [label.decode('utf-8') for label in meta_data[b'label_names']]
    return label_names


# CIFAR-10 label names
cifar10_labels = load_cifar10_meta('./data/cifar10/cifar-10-batches-py')

# Create text descriptions
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar10_labels]).to(device)

# Encode text using CLIP
with torch.no_grad():
    text_features = model.encode_text(text_inputs)

# Compare image features and text features
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Output the top-1 prediction for each image
top_predictions = similarity.argmax(dim=-1)
print(f"Top predictions: {top_predictions}")

# Map predictions to labels
predicted_labels = [cifar10_labels[prediction] for prediction in top_predictions]
ground_truth_labels = [cifar10_labels[label.item()] for label in labels]

# Display results for the first image in the batch
for i in range(len(predicted_labels)):
    print(f"pred:{predicted_labels[i]}, y:{ground_truth_labels[i]}")

