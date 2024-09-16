import torch
import torchvision
import torchvision.transforms as transforms
import clip  # OpenAI CLIP package
from PIL import Image

# Load the CLIP model and the corresponding image preprocessing
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

