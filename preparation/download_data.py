import torch
import torchvision
import torchvision.transforms as transforms

# Define a transformation (you can customize it)
transform = transforms.Compose([transforms.ToTensor()])

# Download CIFAR-10 training data
trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)

# Download CIFAR-10 test data
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

# Create DataLoader to iterate over the data
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

