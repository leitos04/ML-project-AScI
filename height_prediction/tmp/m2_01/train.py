import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SPMDataset
from models import HeightPrediction
import contextlib


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

#---------------------------- Data Loading ---------------------------- 

hdf5_path = '/l/dsh_homo.hdf5'

train_loader = DataLoader(SPMDataset(hdf5_path=hdf5_path,
                                    mode='train',
                                    scan='stm',
                                    height='random'),
                          batch_size=30,
                          shuffle=True)

val_loader = DataLoader(SPMDataset(hdf5_path=hdf5_path,
                                    mode='val',
                                    scan='stm',
                                    height='random',),
                          batch_size=30,
                          shuffle=True)

X, h = next(iter(train_loader))
X = X.to(device)
h = h.to(device)

print(f"Train Loader: {len(train_loader)*30} images. {len(train_loader)} batches of 30.")  # 18000 stm images
print(f"Test Loader: {len(val_loader)*30} images. {len(val_loader)} batches of 30.") #20010 stm images

print("X.shape: ", X.shape)  # X: [N, C, nx, ny] Random slice from the STM scan
print("h.shape", h.shape)    # h: [N]            Height of the random slice

net = HeightPrediction().to(device)

#----------------------------- Training Loop ---------------------------- 

def test_Net_shapes():

    # Feed a batch of images from the training data to test the network
    with torch.no_grad():
        X, h = next(iter(train_loader))
        X = X.to(device)
        print('Shape of the input tensor:', X.shape)

        y = net(X)
        assert y.shape == torch.Size([train_loader.batch_size, 1]), f"Bad y.shape: {y.shape}"

    print('Success')

test_Net_shapes()

#----------------------------- Training Loop ---------------------------- 

mse_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train(dataloader, model, loss, optimizer):
    model.train()
    for batch, (X, h) in enumerate(train_loader):
        X = X.to(device)
        h = h.unsqueeze(1).to(device)  # Shape of h is now [N, 1]

        optimizer.zero_grad()
        outputs = net(X)
        loss = mse_loss(outputs, h.float())

        loss.backward()
        optimizer.step()

        if batch % 500 == 0:
            loss_value, current = loss.item(), batch+1
            print(f"Loss:{loss_value:>5f} [{current:>4d}/{len(train_loader)}]")
            
#------------------ Accuracy on the test dataset ----------------------------   
    
def compute_accuracy(testloader, net, tolerance):
    net.eval()
    correct = 0
    total = 0
    mse = 0
    with torch.no_grad():
        for X, h in testloader:
            X = X.to(device)
            h = h.unsqueeze(1).to(device)
            outputs = net(X)
            loss = mse_loss(outputs, h.float())
            #correct += (outputs == h).sum().item()
            correct += (torch.abs(outputs - h) <= tolerance).sum().item()
            #mse += mse_loss(outputs, h.float()).item() * h.size(0)
            total += h.size(0)
            loss_list = loss_list.append(loss.item())
            
    #mse /= total
    #print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"\nAccuracy: {(correct/total)*100:>0.1f}%")
    print(f"{correct} correct predictions of {total}")

#--------------------------------- Training ---------------------------- 

epochs = 3
for i in range(epochs):
    print(f"-----------------------------\n        Epoch {i+1} \n-----------------------------")
    train(train_loader, net, mse_loss, optimizer)
    compute_accuracy(val_loader, net, 0.1)

print("DONE")

# ---------------------------- Saving model ---------------------------- 

torch.save(net.state_dict(), 'height_pred.pth')
