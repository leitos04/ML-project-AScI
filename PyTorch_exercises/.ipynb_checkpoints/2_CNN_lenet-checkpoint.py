{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "19fbe744-490a-4ffb-a748-a400ea3ec6d5",
   "metadata": {},
   "source": [
    "# Exercise 2. Convolutional neural networks. LeNet-5.\n",
    "\n",
    "In this exercise, you will train a very simple convolutional neural network used for image classification tasks.\n",
    "\n",
    "If you are not fluent with PyTorch, you may find it useful to look at this tutorial:\n",
    "* [Neural Networks](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9e4545a9-0494-411b-8019-dee593154a01",
   "metadata": {},
   "outputs": [],
   "source": [
    "skip_training = False  # Set this flag to True before validation and submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5bc9fd17-6b7f-42f4-9e0d-91f4477aac8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# During evaluation, this cell sets skip_training to True\n",
    "# skip_training = True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cd9f21be-1b0a-4a06-a2de-eea5242278aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "import torch\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torch.optim as optim\n",
    "\n",
    "from pathlib import Path\n",
    "import tests"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "c758967b-4da4-47f4-9689-7e4458985d11",
   "metadata": {},
   "outputs": [],
   "source": [
    "# When running on your own computer, you can specify the data directory by:\n",
    "data_dir = Path('./2_data/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "41ca88bc-6de4-4ce3-8a0c-7cc81bbca77e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Select the device for training (use GPU if you have one)\n",
    "#device = torch.device('cuda:0')\n",
    "device = torch.device('cpu')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "049d03b8-fcac-4957-97ef-0a29f058236c",
   "metadata": {},
   "outputs": [],
   "source": [
    "if skip_training:\n",
    "    # The models are always evaluated on CPU\n",
    "    device = torch.device(\"cpu\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da7ffb3b-ec97-411b-8022-68dfa4d40562",
   "metadata": {},
   "source": [
    "## FashionMNIST dataset\n",
    "\n",
    "Let us use the FashionMNIST dataset. It consists of 60,000 training images of 10 classes: 'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "f34c2b1a-63c6-4fba-b9cf-f602924cd229",
   "metadata": {},
   "outputs": [],
   "source": [
    "transform = transforms.Compose([\n",
    "    transforms.ToTensor(),  # Transform to tensor\n",
    "    transforms.Normalize((0.5,), (0.5,))  # Scale images to [-1, 1]\n",
    "])\n",
    "\n",
    "trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform) #loading training data \n",
    "testset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform) #loading test data\n",
    "\n",
    "classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']\n",
    "\n",
    "#We pass the Dataset as an argument to DataLoader.\n",
    "trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True) # This wraps an iterable over our dataset, and supports automatic batching data loading.\n",
    "# There are 1875 mini batches of size 32.\n",
    "testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=True)\n",
    "# There are 2000 mini batches of size 5."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfec6590-8de7-4336-8bb9-3b8d2786a3ba",
   "metadata": {
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "b3eec87e2b4206e1a149c9169348fcc3",
     "grade": false,
     "grade_id": "cell-a8894f680446eafa",
     "locked": true,
     "schema_version": 3,
     "solution": false
    }
   },
   "source": [
    "Let us visualize the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "eeba9d06-65fe-48b9-9360-fd9ae7f5120d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# print(images[0].ndim)\n",
    "# print(images[0].shape)\n",
    "# print(images[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "df390f2f-18d1-4345-a179-d59458fcb4e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1000x1000 with 8 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "images, labels = next(iter(trainloader))\n",
    "\n",
    "def plot_images(images, n_rows):\n",
    "    n_cols = len(images) // n_rows\n",
    "\n",
    "    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 10))\n",
    "    axes = axes.flatten()\n",
    "\n",
    "    for img, ax in zip(images, axes):\n",
    "        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))\n",
    "        ax.axis('off')\n",
    "    plt.subplots_adjust(hspace=-0.65)\n",
    "    plt.show()\n",
    "    \n",
    "plot_images(images[:8], n_rows=2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d55c8e6-d1c4-426e-ba3e-aba3cb4f727e",
   "metadata": {},
   "source": [
    "# 1. Simple convolutional network\n",
    "\n",
    "In the first exercise, your task is to create a convolutional neural network with the architecture inspired by the classical LeNet-5 [(LeCun et al., 1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c9666e4b-8e31-49bf-a129-575d6c78048f",
   "metadata": {},
   "source": [
    "The architecture of the convolutional network that you need to create:\n",
    "* 2d convolutional layer with:\n",
    "    * one input channel\n",
    "    * 6 output channels\n",
    "    * kernel size 5 (no padding)\n",
    "    * followed by ReLU\n",
    "* Max-pooling layer with kernel size 2 and stride 2\n",
    "* 2d convolutional layer with:\n",
    "    * 16 output channels\n",
    "    * kernel size 5 (no padding)\n",
    "    * followed by ReLU\n",
    "* Max-pooling layer with kernel size 2 and stride 2\n",
    "* A fully-connected layer with:\n",
    "    * 120 outputs\n",
    "    * followed by ReLU\n",
    "* A fully-connected layer with:\n",
    "    * 84 outputs\n",
    "    * followed by ReLU\n",
    "* A fully-connected layer with 10 outputs and without nonlinearity."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "32feaddd-c7e1-486f-be4e-21cc79c099b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "class LeNet5(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(LeNet5, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)\n",
    "        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)\n",
    "        self.fc1 = nn.Linear(16*4*4, 120)\n",
    "        self.fc2 = nn.Linear(120, 84)\n",
    "        self.fc3 = nn.Linear(84, 10)\n",
    "                \n",
    "    def forward(self, x):\n",
    "        x = self.conv1(x)\n",
    "        x = self.relu(x)\n",
    "        x = self.maxpool(x)\n",
    "        x = self.conv2(x)\n",
    "        x = self.relu(x)\n",
    "        x = self.maxpool(x)\n",
    "        x = x.view(x.size(0), -1)\n",
    "        x = self.fc1(x)\n",
    "        x = self.relu(x)\n",
    "        x = self.fc2(x)\n",
    "        x = self.relu(x)\n",
    "        x = self.fc3(x)\n",
    "        return x\n",
    "    \n",
    "net = LeNet5()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6922d20c-fbb0-4417-9ca1-5dfdd1f4e35f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Shape of the input tensor: torch.Size([32, 1, 28, 28])\n",
      "Success\n"
     ]
    }
   ],
   "source": [
    "def test_LeNet5_shapes():\n",
    "    # Feed a batch of images from the training data to test the network\n",
    "    with torch.no_grad():\n",
    "        images, labels = next(iter(trainloader))\n",
    "        print('Shape of the input tensor:', images.shape)\n",
    "\n",
    "        y = net(images)\n",
    "        assert y.shape == torch.Size([trainloader.batch_size, 10]), \"Bad shape of y: y.shape={}\".format(y.shape)\n",
    "\n",
    "    print('Success')\n",
    "\n",
    "test_LeNet5_shapes()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "b531c830-ae7d-49d6-8f90-2fd73e83cb00",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[(6,), (6, 1, 5, 5), (10,), (10, 84), (16,), (16, 6, 5, 5), (84,), (84, 120), (120,), (120, 256)]\n",
      "Success\n"
     ]
    }
   ],
   "source": [
    "def test_LeNet5():\n",
    " \n",
    "    # get gradients for parameters in forward path\n",
    "    net.zero_grad()\n",
    "    x = torch.randn(1, 1, 28, 28)\n",
    "    outputs = net(x)\n",
    "    outputs[0,0].backward()\n",
    "    \n",
    "    parameter_shapes = sorted(tuple(p.shape) for p in net.parameters() if p.grad is not None)\n",
    "    print(parameter_shapes)\n",
    "    expected = [(6,), (6, 1, 5, 5), (10,), (10, 84), (16,), (16, 6, 5, 5), (84,), (84, 120), (120,), (120, 256)]\n",
    "    assert parameter_shapes == expected, \"Wrong number of training parameters.\"\n",
    "    \n",
    "    print('Success')\n",
    "\n",
    "test_LeNet5()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "115d9d6b-03d9-4a62-847b-9a893e9ddd6f",
   "metadata": {},
   "source": [
    "# Train the network"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "04a29d16-c6a2-4993-9da0-f19548af8c12",
   "metadata": {
    "editable": false,
    "nbgrader": {
     "cell_type": "markdown",
     "checksum": "64b0742138de54b013ffd324da1f314d",
     "grade": false,
     "grade_id": "cell-6ade8368217a66dd",
     "locked": true,
     "schema_version": 3,
     "solution": false,
     "task": false
    }
   },
   "source": [
    "### Training loop\n",
    "\n",
    "Your task is to implement the training loop. The recommended hyperparameters:\n",
    "* Stochastic Gradient Descent (SGD) optimizer with learning rate 0.001 and momentum 0.9.\n",
    "* Cross-entropy loss. Note that we did not use softmax nonlinearity in the final layer of our network. Therefore, we need to use a loss function with log_softmax implemented, such as [`nn.CrossEntropyLoss`](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss).\n",
    "* Number of epochs: 10. Please use mini-batches produces by `trainloader` defined above.\n",
    "\n",
    "We recommend you to use function `compute_accuracy()` defined above to track the accuracy during training. The test accuracy should be above 0.87."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9573dd31-42b0-4d94-af6a-82e7559178f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "CEL_loss =  nn.CrossEntropyLoss()\n",
    "optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "94aae1e5-fd01-412f-9aa1-af05245e2059",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1875"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(trainloader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "199e6bd3-6963-411f-88e6-28ba7d2431f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train(dataloader, model, loss, optimizer):\n",
    "    for batch, (images, labels) in enumerate(dataloader):\n",
    "        \n",
    "        optimizer.zero_grad()\n",
    "        \n",
    "        outputs = net(images)\n",
    "        loss = CEL_loss(outputs, labels)\n",
    "        \n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        \n",
    "        if batch%300==0:\n",
    "            loss, current = loss.item(), batch+1\n",
    "        \n",
    "            print(f\"Loss:{loss:>5f} [{current:>4d}/{len(trainloader)}]\")\n",
    "            \n",
    "# train(trainloader, net, CEL_loss, optimizer) #Check the loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "18dbabc3-e971-4481-9da4-977124ad1734",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This function computes the accuracy on the test dataset\n",
    "def compute_accuracy(testloader, net):\n",
    "    net.eval()\n",
    "    correct = 0\n",
    "    total = 0\n",
    "    with torch.no_grad():\n",
    "        for images, labels in testloader:\n",
    "            # images, labels = images.to(device), labels.to(device)\n",
    "            outputs = net(images)\n",
    "            _, predicted = torch.max(outputs.data, 1)\n",
    "            # print(predicted)\n",
    "            total += labels.size(0)\n",
    "            correct += (predicted == labels).sum().item()\n",
    "    print(f\"Accuracy: {(correct/total)*100:>0.1f}%\")\n",
    "    return correct / total"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fde3d906-c50c-400a-b485-dcd6cd8bd3ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2000"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(testloader)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "86407310-f17d-4e14-9446-9ee7a0852ce5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1 \n",
      "---------------------\n",
      "Loss:0.330328 [   1/1875]\n",
      "Loss:0.122480 [ 301/1875]\n",
      "Loss:0.294109 [ 601/1875]\n",
      "Loss:0.258523 [ 901/1875]\n",
      "Loss:0.510316 [1201/1875]\n",
      "Loss:0.403609 [1501/1875]\n",
      "Loss:0.178428 [1801/1875]\n",
      "Accuracy: 87.4%\n",
      "Epoch 2 \n",
      "---------------------\n",
      "Loss:0.310344 [   1/1875]\n",
      "Loss:0.100019 [ 301/1875]\n",
      "Loss:0.211378 [ 601/1875]\n",
      "Loss:0.275836 [ 901/1875]\n",
      "Loss:0.158837 [1201/1875]\n",
      "Loss:0.340737 [1501/1875]\n",
      "Loss:0.313951 [1801/1875]\n",
      "Accuracy: 87.9%\n",
      "Epoch 3 \n",
      "---------------------\n",
      "Loss:0.293473 [   1/1875]\n",
      "Loss:0.359381 [ 301/1875]\n",
      "Loss:0.181791 [ 601/1875]\n",
      "Loss:0.249769 [ 901/1875]\n",
      "Loss:0.405732 [1201/1875]\n",
      "Loss:0.355164 [1501/1875]\n",
      "Loss:0.108868 [1801/1875]\n",
      "Accuracy: 87.6%\n",
      "Epoch 4 \n",
      "---------------------\n",
      "Loss:0.233774 [   1/1875]\n",
      "Loss:0.253283 [ 301/1875]\n",
      "Loss:0.281768 [ 601/1875]\n",
      "Loss:0.409076 [ 901/1875]\n",
      "Loss:0.324070 [1201/1875]\n",
      "Loss:0.188872 [1501/1875]\n",
      "Loss:0.416881 [1801/1875]\n",
      "Accuracy: 87.7%\n",
      "Epoch 5 \n",
      "---------------------\n",
      "Loss:0.397974 [   1/1875]\n",
      "Loss:0.321155 [ 301/1875]\n",
      "Loss:0.453358 [ 601/1875]\n",
      "Loss:0.218189 [ 901/1875]\n",
      "Loss:0.292313 [1201/1875]\n",
      "Loss:0.166533 [1501/1875]\n",
      "Loss:0.128462 [1801/1875]\n",
      "Accuracy: 88.5%\n",
      "DONE\n"
     ]
    }
   ],
   "source": [
    "epochs=5\n",
    "for i in range(epochs):\n",
    "    print(f\"Epoch {i+1} \\n---------------------\")\n",
    "    train(trainloader, net, CEL_loss, optimizer)\n",
    "    compute_accuracy(testloader, net)\n",
    "    \n",
    "print(\"DONE\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "9f70fa0e-b391-4e57-9ae5-43f82b1a0602",
   "metadata": {},
   "outputs": [],
   "source": [
    "if not skip_training:\n",
    "    torch.save(net.state_dict(), '2_lenet5.pth')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "fbbf9d45-569c-42a8-a96f-3b95b784b8e1",
   "metadata": {},
   "outputs": [],
   "source": [
    "if skip_training:\n",
    "    net = LeNet5()\n",
    "    model.load_state_dict(torch.load('2_lenet5.pth'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "7f437183-c4fe-48f8-878b-4ecbff0e566e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAxsAAACZCAYAAABHTieHAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAmF0lEQVR4nO3daZAcZ53n8aeyzj6qL3WrW7Iuy7osWbbwId8GPD7A2IAZzomAGWB2GI7dWdggZpjLEQwRMMsCczHGLGBgh2OZYVkzGMyxwRqDLd/yIcs6LNk6WmdLfXfdtS949Ty/v7eTlpJuqb+fd/mPp7Kyq7Iy6+nKX/5TzWaz6QAAAADgNItmewMAAAAAnJ2YbAAAAABIBJMNAAAAAIlgsgEAAAAgEUw2AAAAACSCyQYAAACARDDZAAAAAJAIJhsAAAAAEpGJO/DG6C1JbsfpFaW11qhP+7ALHte5121dW6V2sNYttQXpcan9xaff4y333vWQPqm1rc2GUZubvRd/2vjX38rzzNX9r3rTpVI78r4pqf3J+p9LLe389/nT/3q7jFn0YE1qhcExqUUntNYYGfXH9C3Qbb1+kdTa33pIah35krd8/M4VMqb47S1SS9pva/9zbu7ug5hd8/0YaDn6oauk1nbrYakN7u7zlqMpPQc3s3ruaxSsc+T027Vx/T6prWw/LrX7v7JZags//+D0TzAL2P8wm+Luf/yyAQAAACARTDYAAAAAJILJBgAAAIBEMNkAAAAAkIjYAfHTKpXSmhGATmX8zWvWNCxrihEGd865XV+72Fv+8aKvyJiRhoZ9ry5UpZZNadD7xr/+vLd8y10Xy5i42xqKWlt1VVO6rXM1WD7nxNgnj7/vShlyz198Wmo7qp1S21UekFrD+fvMw+/5rIzp/MMW3a4Z+vpor9TeUTxibJeGL3dX/c/ehs/qdt37iYLU/mHVut9kEwGcBV757kek9s4eDVhX1/jHwCsKeh49Wp+Q2jOVDqn9ToueS7eU/NpEMydjuiI9b25/kx6v3ee1BCAeftkAAAAAkAgmGwAAAAASwWQDAAAAQCKYbAAAAABIxOwExK3QshHQjRMIT3doUGz7pzSU+r3X/qPUlmf8wNq+mgZjeyMNlFn2VrWDeH/af+wNz2qX52/cdbPUBu7UcF34WjQmJ2NtV9ww/rwSs8N8Zsk53vLjd9wpYz5w8Aap3dT1rNTeVNwptaG6/97cP6UB7nRK98lSQ/fJlyr62Mtbd3vLbytqZ3CjD6/70shKqRWDEOUPxzUE/9GeF6T2p/9rg9TOedM241lxRon5GQrt/twVUlt75zGp1XfqvoQzxwODegy5qrhLaqVG1lveMtUuY76yS2/MMX5Az/uvvPQ5qb26a/v/dzudc66e0f+5HvzpMqmd4w5Muy4ANn7ZAAAAAJAIJhsAAAAAEsFkAwAAAEAiZiWzETbrc87OZ0Rtbd7y4W8ulTF3X/g1qW3I/VxqJxu6/sG6P9cqGPGGqtPrkNPOGGjYW/Mf+/udT8uYD/6pXlP69If1euhtZT8/8E//9CYZs/Dz2jTJzGfM8Hrrs0bMv7XRU/SWb7nxbTKmvm2H1O666DapffyqLqlNDvj7UT0fL0sTGZvf+5Q+9mfPXuqvvz0vY3b+oTbi2/26u6SWTvmflS+OaG7kyyPaCOvJy78utVvdJVLD3BX3eG156TsbveX3rtNjc/ZGXdfdz+t1+kvfrFmo0KlsK06fTX2DscY1gv93rs4fljGf2/gdqR07XzMbA5kRqR2t+8fwMCPyctLlWMMAxMQvGwAAAAASwWQDAAAAQCKYbAAAAABIBJMNAAAAAImYlYB43MDe5geHveW/7P2FjHmyovOlh8oagC6kNNCadn6otto0mvvUNVTbMMa1GtO2XNAybU9Vt6HidFs7UppOu7HNb9D2rj/XJoWX3/x7Uut7vQaY51UY3BAVNBQd9S6QWu2Z4LWL2Qyx8ZSG/vuemv5x6VXnSq2+e2+s5zStXeUvP/KMDFmj/SPdLe5iqe399oXe8s7rNPj9jyeXSy0ybqbQvOoiqaUejPECYVakWlqk1hzTBqWpSy+Q2sXn+I3Q7ju0Xsa8ZckTUvubC78vtQUv+I1TP3nehTLGOrfMODRu3UgDsbSkq1KbbOi5NHS42iW1QqTraov0HLmv2iO1XMo/11Wbui+EIXXnnBtbqedIvf0FgLj4ZQMAAABAIphsAAAAAEgEkw0AAAAAiWCyAQAAACARsxIQt5Rfd5nUPrrAD0H/fKooYxakJ6RWSGmgLAxrW6KUBoDTTX1cGDr79XNqrdFMBWM0lFhw8cLygzU/pPl8RUPOD1/yTald/Qcfklr3Vx+K9ZxnKysM3hzVwGt6YZ9fqOh+1ZiclJoZSC0bLWmDrtypqu4LaWNb68eHpJZZco7UXLg+I/Ca7tROvPWTJ6V27tuf9gtGg+BiekpqDaefqaELWqXW+6CuD7MgpYH+hhEGt+z/mL7Xt3X7N7YYKep7v2NSo7effexmqX3uBv/4lu5fKGPqR45KbcYdxOf5jTROxTUdO6W23whwt0YVb7krrcfTtHHutjqBp43zd2sQJB+u6/53tKbfK95yzcNS2yoVAHHxywYAAACARDDZAAAAAJAIJhsAAAAAEsFkAwAAAEAi5kxAfN/NOu+JgrmQ1Uk07AL+cupGJ+M4j80awe+4zzlTVlA9CkJyHVFJxjSMIF3nOw9IzX11xpt2xknlja61aWOOHen+4aaC1zinocRUTrvCu6rup1YXZtfw36/afk1dp7L6Ec0sitfLtn7wkL8u429sVipSszqsN0r+a3HLjltkzGfO/TepfW9ikdSGz9f9u1cqmA2pjO7jzaruIwc+dpXU/uqCb0nt8YkV3vJl7XtkzPXtz0ntHTfrTSyeLy/2lssblsqYjBEQn6ndf3eF1Nb82dbTtv6zRWblCqkdq+lNJibreiwuB0HvRlOPzXnjvN8R6c0oLIPVbm/5RK1dtyvS7Sqm9fwatfoBd+vmIABs/LIBAAAAIBFMNgAAAAAkgskGAAAAgETMmczGwlXaqKzcDJsx6fXxVhZjpvmMsAnfr59x+maAL/dYK3sR53FWrS1oCDhhvHVjDb22+vUDT0vtB65bamcrM1NR1/c01dYmtWbYxM+4dj1l5T/SRk4kZYwLGkZGkTacstbfNDIhzZPD+tiwQZuVGzGkCsb2B5mNPb9YIUPWrNGsx2B9XGrfv/1zUvvIh6+MtW04zYJ9xMpnWMo9emz75PbXSO2Da+/3ln98YqOMeWFUEzvruw9LbVWLn8f4zJf/WcaUmnpc/JchzZf8cPsGqRUf9/ffVJ/RpHLTGqnNd4du1lxWPnpEaqM1PT60Z/yme/vK2viv2tRmpBYrXxnWrHPyuJEl6c5qs+DU4n6/sHtvrO0CwC8bAAAAABLCZAMAAABAIphsAAAAAEgEkw0AAAAAiZgzAfEbFu+QWjYI1eacFQDTsG89ZqAsDJKfSrM+K3gWBr2t4LrFDsSFYXk12dRtuLZ1p9R+4C6PtR1ng6ijKLVmSRs2pTLGR6G7wx8zqmFna13OaCRYj9FszGpA2CiXjZEq3dWp2xYE4c0we8PY561aYOnPNDyb/g+6/iemVkjtX750s9QG3IPTPidOv1TaP9Y0a3qcSW9YK7W/v/1uqX1ytzZ6bIv8/ffG7m0y5pYePa7vKGno+FDF38f/b0q3yzqG39T1jNRuuFK3o+ca//P9xcOvkjHHtgxLbb5b9ONDUjv0x11Ss85rQxW/yV4m0n3BkjZu3DJVN24GkvZveFBt6DZ0ZvRY9uTIMqnVCYTPT5HxfawRbz899BH/5hTj5+njVn/w4Zlth7UN4U1hXo7xXTH8DtSsG+s3HhcXv2wAAAAASASTDQAAAACJYLIBAAAAIBFMNgAAAAAkYs4ExM/NH5PaZNMPqBRihl/MgLXRXXSmgfA4ncEtcQPihZSGNMMgfN4Iy5eMzuObjNDxvJI3QoNjRtC7X7sY1zr9jreZE8O6LiNYbobBr7hQSqU+f/2Ff9euu+VbLpNaywPP63MOj0gtavU7ktfHxnSM0Tk9DA0751xUCLr/PvCkjLl58SapWQiDzx1Ru//+W/tRo1U/Q+uzx6X2hiVPSS0bHMsKqaqMyRnH5rd1Pi618Pg5WNObP0w0dVuP1TqktrvUL7VbO7Z6yy99WgPorS5mmHMeqe15UWqrCkekNl7X0LV0Fdfct6kS86tLGBqPjBvKtEYVqb1mgd5U4BtuSbyNw9klZhj85O9fKbX89f5x8o4198mYT9yjN9ZY+AY9x8t2WN+HTyHAbd0c5HTilw0AAAAAiWCyAQAAACARTDYAAAAAJILJBgAAAIBEzJmA+I1tu6U2FmS5etMaLrRUmzqHihvOTpIVSLdqVgC9GHRWtWaJ5ZjZoPSCHqnVh07Ee/CZpqqhp8bkpNSs17MZ+fvM1KUrZUzuvkelVrp1s9SGV+tHreWov4PXf9fo7G7stofftVFqvU9rF9wwxG2974ffoiHYvi88pOsKnNI+dJqDbZg5KxAe2vFHBand/M2PSq1z45DU/mbdPd7ywWq3jFmQ0Rs2PFMZkFqpkfWWC5GeD0bruq2WQ6VOqRW7/KBw4YQGhxHP946+QmqbOg9ILQyIWwHuhnE+j2Le3CXsSN4wbqKSN/ajB0b0uOjcRKznxBxwOs8xm/V8u/P9eiOK1Ljuu7knFnjLHxt/o67ruq9Lbf1ffkBqSz8R3FjlFM6Z+/76Kqllg8Pwos+e3hu58MsGAAAAgEQw2QAAAACQCCYbAAAAABLBZAMAAABAIuZMQHxZpl1q2yp+6LUz0gDO5CmEZGbaQfx0Pp8VBp8MgpDOOdcV+UHnnBGAGmvE7LB+vnZyjX55dgbEG8e003HK6qre0H0retDvIpvbvEHHXLBOahMD2oG7/aCuPzvmhxfzQyUZM7pKO3z3bC9L7aXXajD2vK1+h+X6SQ0Dv/s//lBq9/1Iu503J/xQff24hoFjh/IIg8+KVFZDjc2qH4I+/j7tgnvHtd+V2q9GVkvt7b1bpDZU94/rdeP/W5HRNnq43iq1MEh+sKo3KRjIDEtte+kcqbVl9DMU3oTj8r9/TMY8ukk/25i5dHD+s8LgiW+DcV4eqxnnCALic9NpDIMf/DMNTk+sMDpr66naLVp9TGr1hr8/L27Xc/A3xhZIbcVdu3Rd+pSx7PmUHtPrA3r8y3T637cbj2ySMdEvt85wK/hlAwAAAEBCmGwAAAAASASTDQAAAACJmDOZDUucRnyH69a1lSpOXsJqBtiaMprCGdsVp5Gg9ffkmkZWwGhspNul1w4Pp+Jd1TdyXovUun8Z66FnnEbJuLjSusZzu14j6SL/NY7GtclXvVOzEt07jOzFuTouG1wCfOwSzS0V9+t7Wu3Q937Bs7p/l6728yRWA0KrodWdD3xLah956Y3e8ti1MoQsxlwS6T4S5jMsS39vj9R2TmmDvSWFk1IL8xnOObcs42fBwsZ8zjm3p7JQalnjWHYkaMS3ODssY7rS2rCzO6PX2m8bXyS1n06s8pava39exjyaukBqUItb9Nr0OI34wiZ8L6fWOH3ZGeu8PFAYlZru8TCFxx7jO04qp/kxVzfe+7S/rmZZswaxzztGc749b/aPWalGvHWtXXtQar0FPc6ETSTf1a+N8n4xpg0kO/63vhZP7PezlNFOzXOWB/T76ute8YTUdo/1Su2lIT8Dt/8m/Z64/BS+J/LLBgAAAIBEMNkAAAAAkAgmGwAAAAASwWQDAAAAQCJmJSCeXnOeUd067eOKkYaKCkaQ0GqKl44RnraCYlYYPE5w3RK3qV+jMbM5YL0Zb7uOX6zP2f21GT3l3HcqDX8a/j4TjYzLkPHVnVIbW2KEF43NqLX4H7/+X2kEcXStrn+yX9efP6khvJaXhr1l6xNwqNIlta0ZDey2Zvxw8ZixLpz51nYckdqinIZ9S009dYzVNVB4OOXvvyty2mTz6vSLUrOOseHx86ARSL9vRBtSZo3Q8eWde6U2EjQStMLmmeVLpQbVkdGbZFSbetzKBOfltNHg0WoEOVPWjQfKxveFdS2HpLbd9Z227ThrWOfXxvTftcygt6VmNNQL13X1Jqnt/x1tClpaoTfIKPYMe8utOb1hSm+rBr+rdd2XV7cdldr5LX6Q3Gpgel1xh9Te3v2I1AaW+a9r6zW6DU9V9DvyvSObpHZsQsPlmYy//vG+mbYRtPHLBgAAAIBEMNkAAAAAkAgmGwAAAAASwWQDAAAAQCJmJSA+uapn+kGGalMDK1ZAfMxpSEYjYNrdMWlWQNyqWeHIcFZYj9GN9eVkx367f/esOo1drZt53YvKHUbX+aMaAqsV9DVvP+gH1g5fo5+LghH8towt1+1YcL8GXENv7XxMao+Wlknt4o593vKPXFes7TqlgD5mLBXp62408nWZgX5vuZh+ScZYXeaNnKPpYLXbWx6ua3Czmo3XDbotFXxeanrzhPa0Bk+X5zWUviCtN3vYVfY7padT+kdWF3dLDWpRblhq1s0oQqcSBrdu5mIFzkOTDf2+sGdKOyw7NzWTzTq7nc5j+RV6c4fB6/ybQExcoDceOHexfr5Twxosb4t0W8NA+ECb3vpkWdsJqVk3GrAM1/0gdlukx6d6U/f5745cIrX2tP+3t0YaeP/8c6+U2tSQ3rgjlTe6umf8Wv9y/btPBb9sAAAAAEgEkw0AAAAAiWCyAQAAACARTDYAAAAAJGJWAuIjK6y4tgrD0/mUBrmsvshVo7ttW0pDjmH37nTMsJMV6i45DTmG46zgt/VKWOsfC7qK96X1b5yMuf1G7hExgsypqobO0hV9zTt2a8js0LUaZi135b1lowGzqxb1/wF9WzUY1jQCwbUDfvdSF+k+en5OA7vPVPSz8v7iLm/5R+4yGWMiDJ64qFCQWqOkQUpL53f9fWnvpAZjuzu0g+6t7duktquq4enBml+baORlTBgid865SWNcGB5+VetOGXN9y6DUHivrjRescHkUHBjf2KYh8k+ep58XKKv7+iHjphJyDj6FDuJhN3KLtS4roDuQH5Xai+bZen7b+YXNUhtYPuQtj03p8amrVcP2w5P6/k0M+8en/F5d1+ALS6T2B2/+qdT6syNS+/gjt3rLLVk9900V9Hvn9nE9pvzuoieklkv53xmemlwea7uubNsttdCHn3yr1KJIPz+tvfpZnBzRLxvNcX//rnfp5yLdu2Da7Xo5/LIBAAAAIBFMNgAAAAAkgskGAAAAgEQw2QAAAACQiFkJiJf64o2LgqB0uanhncmG/glh0M8551ojDfdaj52phhEyC8NBVkDc7BZuda4N5oWRMU9MO/0bLdF6DTDPe3GCzHV9X/KjGmo7dkmH1EbX6ntT3OXvf7lh3QYr+D20QQNrS+49KjXX5Ydg68MaRLN0RRoou2fCDw6nuzRga66fDuKn1amEwU/eu1pqm1r94H8xres6WWuT2g/GN0jNCo3f1Oofsw/VNHS9v65h8FJTw7gn6u1SCz1V0c+e1bW8GOnfWTFuLCLb1WPszxDnZfV4tHVimdTidGKuG4cL6xwZhs0t1vOVje8Bn+h/RGq3d93gb1fM4+nZYui9V0rtE6/+jtS2Ty32lq3X/FilKLVyt74P2aX+Y4fXarDZet93TvRLrdqqN0h5/YanveXvP6tdzNd06r58Ve8eqR2p6jkxHeyny3La7dy6mYJ1I41vHfHD+KURHdPVp8fXUkWPpS0dxvGv7I+7cmCvjNl57jqpxcUvGwAAAAASwWQDAAAAQCKYbAAAAABIxOxkNgbiZQsKwfVuLSm9XrlkXP9dMBr4VY1MRdLCPIbVrK/R1GuArXFtQf4jcnrdvnXtYr2p17b2tOs1gphes1OvGa8U9TrQel7f054ndVx+1H9vxs/RfTQzES/HUVrepbVNfs6i+D+3yJgvjiyW2ob8AakVIv8zdfyN62VM91cfkloqrX93sxbv839GMDIpUV6vpQ01jeOW9bj6qN9cLG4+Y9fXLpba7QNbpXa87O/TxVZd/2uLT0vt2ycvl9qnJm6U2oXt/r70q5OrZMzfLPl3qdWdNq4sNfxrirdVBqYd45xzOeOacSsTYh2LQzFiAfNOVNTr75dktGlbtWkcA4MspX0+NLrQpqzM4vQ5DuscaWU2LNWLVvrruv/JWI87W7Qf1uP2sZq+94tyw95yX0YbJF7aVpbaUE3Pr4eChp/L9Sug685o01ErZ7ZtfJHUNhb9xrevWf+cjMkZed+bis9I7cWqhpHrwf7cFulx7XCtS2rWZ6WY9V+zP778fhljNQi0smgL0prt2FPxt39/SRsXuq07tBYTv2wAAAAASASTDQAAAACJYLIBAAAAIBFMNgAAAAAkYlYC4q6ggT0ryJwOsmJTTQ3X7Ktp85aNuUNSKxuBmzAsFoZ5fhNWkyEr6B2ymvoVjEBjnFlhMaVBpobTIORAmwa2aPOn0h1Bg7CK3niguEfDaaVuDboVjIZ9g68O9pmCrj+/X28EsPIb2mTo+BUaTut5zg+BWXvjJ++/VWoP3PJZqT1bWeAtT9ym+1D3V3X9Z3sY3GpQGDfEHaqXNTQZilq1Qd21W4ak1nLyoNQOlbTp1E0L/EZ81s01dlX0GNud0ZtMWLVi5AeFr+veKWMOGs36/sexq6V2WYffZKojskLIekqzamGzrbiMQ+y8N/nK86WWdj+SWq2h5+DWMDBrBL/NG5804o2LwwqWjzT0u8axi/yGcv2azz2r5e99VGpfX/paqQ287SVvebSsqe72nB7rLurWY9bmNr953rrcERnTn9b3rzvS5n/j3Xqji/aUf1OOdM8LMub/TOl++3xZb6xy//AaqW15aYW3XB/U43e6YtwUYcq48UhwaH6ocIGMKRw3vk8O6euTnTJurDTk7/PVNj1utqQ1GB8Xv2wAAAAASASTDQAAAACJYLIBAAAAIBFMNgAAAAAkYnYC4pV4c5xiEBa7Z+IcGfPw2EqpvWHRcantqGroOgxwN4wIrdV53AqiWaHuOB1prYD4qqy+Lf9l8FXe8oXt+2XMezu087MVqh8oaBx8XgXEYwZ9Xfg+GO9L5qCGc6MLNfDa9cig1E6sW+Itd+/Ufe3kOt3W+o7dUmtb0SW11PMvestWfHLgfn3Oxi06brThh/w+d9F3ZMxn3AbjGQxxX/+5JuY2pled6y2Pb9DwfqlLQ4eTA0Zn+IV+uO+uN/53GfO1oxqm3tz9otQOlLulVg+Obz8fXSdjXtG+T2pLcrrfh13mnXPukrwf+rz75JUy5vtHLpLa2xc9Mu36JxvacT1r3iRD9/Hw73bODpKHWoyw5Xx39FJ93Y4ZXbnjduqOI+w87ly8DuLWOXnC2I8Ga7qto6v9c7zeNmH+6fvCQ1Jr3B28nrdtkjFH+/Tz9+PMcqn9pHmVt2x8bF1NM9eu0qnH6lq78dlt+PtDxghm509ozWha7rp3aOi9+mb/OH/n6++WMcN1/QPeXjwptZU/e4+33H+f3jxm5Dzd1kqHcfxr0den0uvv88V+7TLeW9NQelz8sgEAAAAgEUw2AAAAACSCyQYAAACARDDZAAAAAJCI2QmIx5zitEZ+9+uHRlfJmHufvFBq/3Crdro8bKSIBjIzi0VbITMr6B3WzBC50Qk1n9Lw6IOHVnjLDzQ0GP9Hm78ltWpDQ1GjNQ3EOTd99+L5JlXwQ9FWPLh+XIOyC76kYfCG0fm5e6ffhbT7AQ3iRrWlUsss0RsluB8/JqVm3nqffT0PGcH1unadbwTJvMvyI9Ouez7YeedmqV276Xm/UNLP1pKcdr+2bjzx2AH//f+vL2rH3guNzru9WT229Wf1PRsJwol9OQ0F3tq+Q2oFK+Rv+LshPxB+sNQlY87vOCy1FTm9ycfhmt8B3Qp5ZyM9xrqmHgPrVmjcOIaHMmUC4qHG+brPPFdeJDUr1B2yQt7We2WZaQfxlrTeRGVndaHUlqw5OqP1zzfNsn+8a/u3h2VMm/G4zIplUiut9G+uUe3Q70Zlo1bQ07JrpK2gtP+Zt+5hYHxtc5UOrQ1doOfb4i5/+T+deI+MaTmmx52P6z1mXFtwGqkVdH9vHdRaWu/b4XLj+jlLB6ek/I+26wNPAb9sAAAAAEgEkw0AAAAAiWCyAQAAACARs5PZMK6tHG/qdc2dUYu3vG14QFcVs0Fg1el1fVbDvlA2pde2ZY3rShsxMhtW1qPU1OvjnbF+WfcWbdDl9PJx02ilxarGe/DZIGZztsYC/8LM1KTuo1G7cfVpV6eU9r5f80ZRcKnw+DkrdIxxveXh/6w5jrWf0QtLa0eO6YPDMS9qTuTpsmZCutKT3vIXh7URW2TkUhqTk1I7Ixr4GdIdeqHumy/XfFh4bfqJsu4jR6aKUtuzT68Tb+vSbEeo1bjm3FIML8p1zvVl/M99I6/HxOcqeqyxjltW87xq0z/uVht6HD5W0QuU//nQ9VI7v93PdlzQok1Mw+f79bbqtdRp47geR+vB6d+P+WbjYs19hfka55zLWHmaGKwch/U+W+d467EhK0tyoLJAalcu3Ostb512zfhNWOeiTFCzvrBa32Yw9/DLBgAAAIBEMNkAAAAAkAgmGwAAAAASwWQDAAAAQCJmJyBe16B0I0Zo9MCTi7XYH68ZndkAKggJpo22bXEaPTlnN+yrBM9ZNgJsFSPoVm9qYG1Jp9+Q6/heIyBuGDaa+u0b6ZJanzsSa33zSTPrvzcpq5FZpO+fq+u+sPLLGn5rVoJgb1n35fqoNszqN5qUNYwGflHOD/E2a7r9zZrua9umlkjtHV1+Y6Zci/6Nv+hbp9v1khEQt16zxszCo79NozedL7UjZW18lA+CsJd063vfGt4dwDl3wYqfSG1F5uS021U0grfPG6HujqgktfD4Vm3qKeFoXcPsD46tltrhkgboe3L++9+V1YD1AqORYD6l++XRqr8dQ3UNlg9khqVmBeMLKb3zQiX424/XJ2RMI8v/50KLW7RZZLmhNxDozOj7EJ6XC9YdMQzW+TwO60YukXHetwLoPZlwf7Ba0wGwcOQEAAAAkAgmGwAAAAASwWQDAAAAQCKYbAAAAABIxKwExIv9Ggg8YQSZu4OMVpdmMd3xDv0T6kaAdkX2hNTCQHjJCIVZHcSrxhytbHTGjYJO6a1Gp9JzUlbnbg37PrPT7xq9dueY8Tg1afxNC9v19T8zezonLAreZ+MmBqmMvr7Nkr7PjWENUYYB8cgIeac7NXRrBdBd1vgoN4LtndKAphUQv73zcaltmVrpLW8s7Nd1jWug1mR8Ps8ERzbr5/629kNSe+iE/1qdrGiP2+6cvheHKtp1OeyAHB5TnHNuWV6PbZ1pfS8OO11/GAi3OoOHXcadc25jq3bvXpbXDvJhAP1wWbdh76R2a+7P6/Et7PS8Y3JAxuxK9Usta9y8o24cww8FAfeRur5vQxv1b5zvBnK6f8QNcIc3SrA6u0cz7PbunHONYDviblfeCKqP1wsz3g5gvuOXDQAAAACJYLIBAAAAIBFMNgAAAAAkgskGAAAAgETMSkD8+qW7pBZ227ZkyhqOjNo1yJVO6bp6jHB2uDqrW3iX8TjLcENfymwQ7uwx/sRSU8OL1vb3LvIDxumjGvK2WJ3N13QcldqOWGs7A1ldv2N0q3fOuWbaf2xqXLthWwFrl9OQbcqqtQYB1DDQ7Zxz1voj42+q6rhm8HeGy845F7Vq4PWOlZfo+gPfc31GVYPKppiv/1yz+m/1U/KTi7Wr+MU9fni+O6P7jRU+nqrrPtKa9t/XyUZOxjw9rh3fD05qENuSifxjVM443k3W9DlXtOt7bXViDjs2t6XLMmaDEbK3QsFh0NsKfludq61u0NZjw7DyUuOmItxJQ8Xtyh3rscb3AGtfCIPfLyd8rPW+W6wO89Uo3I8IjANx8csGAAAAgEQw2QAAAACQCCYbAAAAABLBZAMAAABAImYlIP7TF9dK7c8X3j/t4yrtGkRrDmt48UBNw9O7q9qJOZvyw5BW99znyvo4q/tsl9GxN+xQPljX8OVwvU1qJ3Ia4D75fI+33Du0T8aUmxpq601rOPLRY8uk1uFekNp81wzD5Vkj5G28vlaHbyucbQbCQ2nj/wFNIyBubIaMMrYr6u7SzZrUQDOcqw9pYDhzg9a2rTrXW97zzkUyZvNNz0rtlp5npPbKFj9sXoz0kG3dZKJk7G8542YJ9WCcHkGc65RgrHNpIxRsbcdksP6Sse/uqfZI7akpPUYdr03fvbsno8fhJTl9j+45tklqoxU/8DvelZcxAz/U4+5898Kk3ixiffug1IbKeq4Lu8Jb6sbNFApGh28rlG51JJf1G2Fz62YxdBAHZo5fNgAAAAAkgskGAAAAgEQw2QAAAACQiFnJbEwe12tvF6b1ek553IBeR7n6Qw9L7b0fumZmGzZnrJTKeW6Lt1y+QRuv5VNbjJrmDDpy2lhrvovadP+rtPgfj6zRYK9ZNa5yj5vPaE5/PbH5OKupXwypjPFxz+j1+FGxqJsxNjaj53TG9f4pY/vN5ohnqPruvd7y8jv2ypgjd+jj7nbLpfb1Va/ylitLu2XMyArNFkz1x9tHrPhPqG1Q98G2I/p+FQZ1H4mOnvSWa4ePxNqumdOMnV0bmnZNT5vVg7/Z5swD7174gNTWZTU781xFjythbtLKQ860gZ9z2pyvaDTrKxiP609rFvREw2/6uMWd6d8zgN8eftkAAAAAkAgmGwAAAAASwWQDAAAAQCKYbAAAAABIxKwExNfdqeGx9fs/ILVaqx9MXHXnLhmjbaTmh+zPHpfapX/1fh1oBEB7Hx+VWtMdOB2bNfdYYW1DY0L3yfxj/v7WtILZWQ0SurSOSxkN1cxGf7JhVtjc2OurRng9DF0bDQjr+7X5lrn+0+hsCoMnLQybp3dr2Fxb4s0O63YHMW6BgDPcn/w3PXePXjUltUbZ6DwaHN5SuZh7TLzDumvWg/+nVvT/q+kJq3GqlloH/XGL3IPxNgIAv2wAAAAASAaTDQAAAACJYLIBAAAAIBFMNgAAAAAkItWMlVIFAAAAgN8Mv2wAAAAASASTDQAAAACJYLIBAAAAIBFMNgAAAAAkgskGAAAAgEQw2QAAAACQCCYbAAAAABLBZAMAAABAIphsAAAAAEjE/wOuxwpU0AosWQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1000x1000 with 5 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Ground truth labels:        Coat T-shirt/top    Sneaker T-shirt/top    Sneaker\n",
      "Predictions:                Coat T-shirt/top Ankle boot T-shirt/top    Sneaker\n"
     ]
    }
   ],
   "source": [
    "# Display random images from the test set, the ground truth labels and the network's predictions\n",
    "net.eval()\n",
    "with torch.no_grad():\n",
    "    images, labels = next(iter(testloader))\n",
    "    plot_images(images[:5], n_rows=1)\n",
    "    # Compute predictions\n",
    "    images = images.to(device)\n",
    "    y = net(images)\n",
    "\n",
    "print('Ground truth labels: ', ' '.join('%10s' % classes[labels[j]] for j in range(5)))\n",
    "print('Predictions:         ', ' '.join('%10s' % classes[j] for j in y.argmax(dim=1)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "d5f36a23-cbb2-43ed-9277-13c3f6fe6d05",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 88.5%\n",
      "Accuracy of the network on the test images: 0.885\n",
      "Success\n"
     ]
    }
   ],
   "source": [
    "# Compute the accuracy on the test set\n",
    "accuracy = compute_accuracy(testloader, net)\n",
    "print('Accuracy of the network on the test images: %.3f' % accuracy)\n",
    "assert accuracy > 0.85, \"Poor accuracy {:.3f}\".format(accuracy)\n",
    "print('Success')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
