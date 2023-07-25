import torch
from torch.utils.data import DataLoader
from data import SPMDataset, normalize
from models import HeightPrediction
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Device:', device)

#---------------------------- Data Loading ---------------------------- 

hdf5_path = '/l/dsh_homo.hdf5'

val_loader = DataLoader(SPMDataset(hdf5_path=hdf5_path,
                                    mode='val',
                                    scan='stm',
                                    height='random',),
                          batch_size=30,
                          shuffle=True)
                          
#---------------------------- Load model ----------------------------
net = HeightPrediction().to(device)

# Load the model weights from the .pth file
net.load_state_dict(torch.load('height_pred.pth'))

net.eval()

X, h = next(iter(val_loader))
#print(X[0])
#print(X.shape)
#print(X.dtype)

with torch.no_grad():
	X_n = normalize(X)	
	X_n = X_n.to(device)
	h = h.to(device)
	outputs = net(X_n)
	outputs = outputs.flatten()
	h_values = h.tolist()
	out_values = outputs.tolist()

"""
print(f"\nTruth Values | Predictions")
for c1, c2 in zip(h_values, out_values):
    #print(c1, c2)
    print("    %.3f" % c1, "        %.3f" % c2)
""" 

#------------- Plot STM images +  write height predictions ---------------

def plot_images(images, subtitles1, subtitles2, n_rows):
    n_cols = len(images) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()
	
    fig.suptitle("Testing simulated STM images - height predictions model 1", fontsize=16)	
	
    for img, subtitle1, subtitle2, ax in zip(images.cpu(), subtitles1, subtitles2, axes):
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.set_title(f"Predicted Height: {subtitle1:.2f}\nTruth Height: {subtitle2:.2f}", fontsize=10)
        ax.axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig("results/simulated_h_predictions_m1.png")
    plt.show()
    

plot_images(X, h_values, out_values, 5)


