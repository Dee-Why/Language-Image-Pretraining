import pickle
import numpy as np
import os

# Function to unpickle a file
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')  # Load in bytes format as required
    return dict_data

# Function to load CIFAR-10 data batches
def load_cifar10_data(data_dir):
    # Initialize lists to hold training data and labels
    train_data = []
    train_labels = []
    
    # Load each batch file
    for i in range(1, 6):  # CIFAR-10 has 5 training batches
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch = unpickle(batch_file)
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    
    # Convert the list to numpy arrays
    train_data = np.concatenate(train_data)
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (num_images, 32, 32, 3)
    
    train_labels = np.array(train_labels)
    
    # Load the test data
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (num_images, 32, 32, 3)
    test_labels = np.array(test_batch[b'labels'])
    
    return train_data, train_labels, test_data, test_labels

def load_cifar10_meta(data_dir):
    meta_file = os.path.join(data_dir, 'batches.meta')
    meta_data = unpickle(meta_file)
    label_names = [label.decode('utf-8') for label in meta_data[b'label_names']]
    return label_names


# Path to the CIFAR-10 data folder
data_dir = './data/cifar10/cifar-10-batches-py'
train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

print(f"Training data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Test labels shape: {test_labels.shape}")


label_names = load_cifar10_meta(data_dir)
print(label_names)  # List of class names

