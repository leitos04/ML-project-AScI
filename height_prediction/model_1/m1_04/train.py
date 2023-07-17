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

mse_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train(dataloader, net, loss, optimizer):
    total_loss_t = 0
    num_batches = len(dataloader)
    loss_list_t = []

    for batch, (X, h) in enumerate(dataloader):
        net.train()
        X = X.to(device)
        h = h.unsqueeze(1).to(device)  # Shape of h is now [N, 1]

        optimizer.zero_grad()
        outputs = net(X)
        loss = mse_loss(outputs, h.float())

        loss.backward()
        optimizer.step()

        total_loss_t += loss.item()
        #batch_mse = total_loss_t / X.shape[0]  # MSE per batch
        #loss_list_t.append(batch_mse)
    
        if batch % 500 == 0:
            loss_value, current = loss.item(), batch+1
            print(f"Loss:{loss_value:>5f} [{current:>4d}/{len(train_loader)}]")
            
    mean_mse_t = total_loss_t / num_batches  # Calculate mean MSE across all batches
    loss_list_t.append(mean_mse_t)
    print(f"\nTraining Mean MSE: {mean_mse_t:.4f}")
    
    return loss_list_t
	
	
#------------------ Accuracy on the test dataset ----------------------------   
    
def compute_accuracy(testloader, net, tolerance):
    net.eval()
    correct = 0
    total = 0
    total_loss_v = 0
    loss_list_v = []
    
    with torch.no_grad():
        for X, h in testloader:
            X = X.to(device)
            h = h.unsqueeze(1).to(device)
            outputs = net(X)
            
            loss = mse_loss(outputs, h.float())      
            total_loss_v += loss.item()
            correct += (torch.abs(outputs - h) <= tolerance).sum().item()
            total += h.size(0)

        mean_mse_v = total_loss_v / len(testloader)   # mean MSE across all batches
        loss_list_v.append(mean_mse_v)
        print(f"\nValidation Mean MSE: {mean_mse_v:.4f}")
        
    print(f"\nAccuracy Test: {(correct/total)*100:>0.1f}%")
    print(f"{correct} correct predictions of {total}")
            
        
    return loss_list_v
            
#--------------------------------- Training ---------------------------- 

epochs = 50
loss_training_list = []
loss_validation_list = []

for epoch in range(epochs):
    print(f"-----------------------------\n        Epoch {epoch + 1} \n-----------------------------")
    training_loss = train(train_loader, net, mse_loss, optimizer)
    validation_loss = compute_accuracy(val_loader, net, 0.1)

    for loss in training_loss:
        loss_training_list.append((epoch + 1, loss))

    for loss in validation_loss:
        loss_validation_list.append((epoch + 1, loss))

    with open("loss_training_values.txt", "w") as file:
    	for epoch, loss in loss_training_list:
        	file.write(f"{epoch}\t{loss:.4f}\n")

    with open("loss_validation_values.txt", "w") as file:
    	for epoch, loss in loss_validation_list:
        	file.write(f"{epoch}\t{loss:.4f}\n")
print("DONE")

# ---------------------------- Saving model ---------------------------- 

torch.save(net.state_dict(), 'height_pred.pth')
