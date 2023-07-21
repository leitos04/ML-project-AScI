import torch
from torch.utils.data import DataLoader
from data import SPMDataset
from models import HeightPrediction

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

#ground_truth = []
#predictions = []

X, h = next(iter(val_loader))
#print(X[0])
#print(X.shape)
#print(X.dtype)

with torch.no_grad():	
	X = X.to(device)
	h = h.to(device)
	outputs = net(X)
	outputs = outputs.flatten()
	h_values = h.tolist()
	out_values = outputs.tolist()

print(f"\nTruth Values | Predictions")
for c1, c2 in zip(h_values, out_values):
    #print(c1, c2)
    print("    %.3f" % c1, "        %.3f" % c2)
   

