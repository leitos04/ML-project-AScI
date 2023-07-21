import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import SPMDataset, normalize
from models import HeightPrediction
import contextlib
import os
import numpy as np

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
                                    height='random'),
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

def train(dataloader, model, loss, optimizer):
    total_mse_t = 0
    total_mae_t = 0

    for batch, (X, h) in enumerate(dataloader):
        model.train()

        X = normalize(X) 
        X = X.to(device)
        h = h.unsqueeze(1).to(device)  # Shape of h is now [N, 1]

        optimizer.zero_grad()
        outputs = model(X)
        
        mse = loss(outputs, h.float())
        mae = torch.mean(torch.abs(outputs - h))

        mse.backward()
        optimizer.step()

        total_mse_t += mse.item()
        total_mae_t += mae.item()

        if batch % 500 == 0:
            mse_value, mae_value, current = mse.item(), mae.item(), batch + 1
            print(f"MSE: {mse_value:>5f} MAE: {mae_value:>5f} [Batch: {current:>4d}/{len(dataloader)}]")

    mean_mse_t = total_mse_t / len(dataloader)  # Calculate mean MSE across all batches
    mean_mae_t = total_mae_t / len(dataloader)  # Calculate mean MAE across all batches
    print(f"\nTraining Mean MSE: {mean_mse_t:.4f}")
    print(f"Training Mean MAE: {mean_mae_t:.4f}")

    return mean_mse_t, mean_mae_t


#------------------ Accuracy on the test dataset ----------------------------   

def compute_accuracy(testloader, model, tolerance):
    model.eval()
    correct = 0
    total = 0
    total_mse_v = 0
    total_mae_v = 0

    with torch.no_grad():
        for X, h in testloader:
            X = X.to(device)
            h = h.unsqueeze(1).to(device)
            outputs = model(X)

            mse = mse_loss(outputs, h.float())
            mae = torch.mean(torch.abs(outputs - h))

            total_mse_v += mse.item()
            total_mae_v += mae.item()
            
            correct += (torch.abs(outputs - h) <= tolerance).sum().item()
            total += h.size(0)

        mean_mse_v = total_mse_v / len(testloader)  # mean MSE across all batches
        mean_mae_v = total_mae_v / len(testloader)  # mean MAE across all batches
        print(f"\nValidation Mean MSE: {mean_mse_v:.4f}")
        print(f"Validation Mean MAE: {mean_mae_v:.4f}")

    print(f"\nAccuracy Test: {(correct / total) * 100:>0.1f}%")
    print(f"{correct} correct predictions of {total}")

    return mean_mse_v, mean_mae_v


#--------------------------------- Training ----------------------------

# Check if the model file exists
if os.path.exists('height_pred.pth'):
    net.load_state_dict(torch.load('height_pred.pth'))
    print(f"Loaded model state from 'height_pred.pth'")
else:
    print(f"Model file 'height_pred.pth' not found. Starting training from scratch.")
    
# Create the "results" folder if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")

epochs = 100
initial_epoch = 0 #---------- Check this!!!

mean_mse_training_list = []
mean_mse_validation_list = []

mean_mae_training_list = []
mean_mae_validation_list = []

best_mse_loss = np.inf
max_consecutive_failures = 7
counter = 0

for epoch in range(epochs):
    print(f"-------------------------------------------------\n                    Epoch {epoch + initial_epoch + 1} \n-------------------------------------------------")
    mse_loss_t, mae_loss_t = train(train_loader, net, mse_loss, optimizer)
    mse_loss_v, mae_loss_v = compute_accuracy(val_loader, net, 0.1)

    mean_mse_training_list.append((epoch + 1, mse_loss_t))
    mean_mse_validation_list.append((epoch + 1, mse_loss_v))
    
    mean_mae_training_list.append((epoch + 1, mae_loss_t))
    mean_mae_validation_list.append((epoch + 1, mae_loss_v))

    with open("results/mse_training_values.txt", "w") as file:
        for e, mse_loss_t in mean_mse_training_list:
            file.write(f"{e+initial_epoch}\t{mse_loss_t:.4f}\n")

    with open("results/mse_validation_values.txt", "w") as file:
        for e, mse_loss_v in mean_mse_validation_list:
            file.write(f"{e+initial_epoch}\t{mse_loss_v:.4f}\n")

    with open("results/mae_training_values.txt", "w") as file:
        for e, mae_loss_t in mean_mae_training_list:
            file.write(f"{e+initial_epoch}\t{mae_loss_t:.4f}\n")

    with open("results/mae_validation_values.txt", "w") as file:
        for e, mae_loss_v in mean_mae_validation_list:
            file.write(f"{e+initial_epoch}\t{mae_loss_v:.4f}\n")
       
        # Check if validation loss has improved
    if mse_loss_t < best_mse_loss:
        best_mse_loss = mse_loss_t
        counter = 0
    else:
        counter += 1
        if counter >= max_consecutive_failures:
            print("Failures of improving validation. Training stopped.")
            break
print("DONE")

# ---------------------------- Saving model ---------------------------- 

torch.save(net.state_dict(), 'height_pred.pth')

