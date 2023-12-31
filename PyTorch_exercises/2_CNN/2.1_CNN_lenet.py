
skip_training = False  # Set this flag to True before validation and submission

import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path

data_dir = Path('./2_data/')

device = torch.device('cpu')

if skip_training:
    device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # Transform to tensor
    transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]
])

trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform) #loading training data 
testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform) #loading test data

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#We pass the Dataset as an argument to DataLoader.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True) # This wraps an iterable over our dataset, and supports automatic batching data loading.
# There are 1875 mini batches of size 32.
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True)
# There are 2000 mini batches of size 5.


# Let us visualize the data.

images, labels = next(iter(trainloader))

def plot_images(images, n_rows):
    n_cols = len(images) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    axes = axes.flatten()

    for img, ax in zip(images, axes):
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.axis('off')
    plt.subplots_adjust(hspace=-0.65)
    plt.show()
    
plot_images(images[:8], n_rows=2)


# 1. Simple convolutional network
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
net = LeNet5()

def test_LeNet5_shapes():
    # Feed a batch of images from the training data to test the network
    with torch.no_grad():
        images, labels = next(iter(trainloader))
        print('Shape of the input tensor:', images.shape)

        y = net(images)
        assert y.shape == torch.Size([trainloader.batch_size, 10]), "Bad shape of y: y.shape={}".format(y.shape)

    print('Success')

test_LeNet5_shapes()

def test_LeNet5():
    # get gradients for parameters in forward path
    net.zero_grad()
    x = torch.randn(1, 1, 28, 28)
    outputs = net(x)
    outputs[0,0].backward()
    
    parameter_shapes = sorted(tuple(p.shape) for p in net.parameters() if p.grad is not None)
    print(parameter_shapes)
    expected = [(6,), (6, 1, 5, 5), (10,), (10, 84), (16,), (16, 6, 5, 5), (84,), (84, 120), (120,), (120, 256)]
    assert parameter_shapes == expected, "Wrong number of training parameters."
    
    print('Success')

test_LeNet5()

# Training loop

CEL_loss =  nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train(dataloader, model, loss, optimizer):
    for batch, (images, labels) in enumerate(dataloader):
        
        optimizer.zero_grad()
        
        outputs = net(images)
        loss = CEL_loss(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        if batch%300==0:
            loss, current = loss.item(), batch+1
        
            print(f"Loss:{loss:>5f} [{current:>4d}/{len(trainloader)}]")
            

# This function computes the accuracy on the test dataset
def compute_accuracy(testloader, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            # images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {(correct/total)*100:>0.1f}%")
    return correct / total


epochs=5
for i in range(epochs):
    print(f"Epoch {i+1} \n---------------------")
    train(trainloader, net, CEL_loss, optimizer)
    compute_accuracy(testloader, net)
    
print("DONE")

if not skip_training:
    torch.save(net.state_dict(), '2_lenet5.pth')

if skip_training:
    net = LeNet5()
    model.load_state_dict(torch.load('2_lenet5.pth'))

# Display random images from the test set, the ground truth labels and the network's predictions
net.eval()
with torch.no_grad():
    images, labels = next(iter(testloader))
    plot_images(images[:5], n_rows=1)
    # Compute predictions
    images = images.to(device)
    y = net(images)

print('Ground truth labels: ', ' '.join('%10s' % classes[labels[j]] for j in range(5)))
print('Predictions:         ', ' '.join('%10s' % classes[j] for j in y.argmax(dim=1)))

# Compute the accuracy on the test set
accuracy = compute_accuracy(testloader, net)
print('Accuracy of the network on the test images: %.3f' % accuracy)
assert accuracy > 0.85, "Poor accuracy {:.3f}".format(accuracy)
print('Success')

