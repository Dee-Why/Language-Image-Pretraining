import torch
import torchvision
import torchvision.transforms as transforms

# Define transformation: Resize images to 224x224 and convert them to tensors
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to a tensor
])

# Download CIFAR-10 dataset with the resizing transform applied
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=False, transform=transform)

# Create DataLoader to iterate over the resized data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Check the shape of the resized images
data_iter = iter(trainloader)
images, labels = next(data_iter)
print(f"Resized image shape: {images.shape}")  # Should print torch.Size([64, 3, 224, 224])

