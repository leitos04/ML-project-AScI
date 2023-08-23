import torch
from torch.utils.data import DataLoader
from data import SPMDataset, normalize
from models import HeightPrediction
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Device:', device)

#---------------------------- Experimental Data Loading-----------------------

data_np = np.load('../sila_resized.npy')
data = torch.from_numpy(data_np).float()

#print(data)
print(data.shape)
#print(data.dtype)

#---------------------------- Load model ----------------------------
net = HeightPrediction().to(device)

# Load the model weights from the .pth file
net.load_state_dict(torch.load('height_pred.pth'))

net.eval()

with torch.no_grad():
        data_n = normalize(data)
        data_n = data_n.to(device)
        output = net(data_n).flatten()
        h_values = output.tolist()

print(f"Height predictions: {h_values}")


#------------- Plot STM images +  write height predictions ---------------

def plot_images(images, subtitles, n_rows):
    n_cols = len(images) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()
	
    fig.suptitle("Experimental STM images: silicon-carbon compound molecule - model 2", fontsize=16)	
	
    for img, subtitle, ax in zip(images.cpu(), subtitles, axes):
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap="afmhot")
        ax.set_title(f"Predicted Height: {subtitle:.4f}", fontsize=12)
        ax.axis('off')
    #plt.subplots_adjust(hspace=-0.7, top=1.2)
    plt.savefig("results/experimental_h_predictions_m2.png")
    plt.show()
    

plot_images(data, h_values, 2)

