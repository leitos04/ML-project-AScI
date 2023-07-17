import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ase import Atoms
from ase.visualize import view
from ase.visualize.plot import plot_atoms

def get_data(hdf5_path, idx, mode='train'):
    """
    Returns an item with index 'idx' from the hdf5 file.
    The file is divided into training, validation and testing data sets.
    
    Arguments:
        hdf5_path: path to the hdf5 file
        idx: index of the wanted item
        mode: which data set should be used options: ['train', 'val', 'test']
    """
    with h5py.File(hdf5_path, 'r') as f:
        data = f[mode]
        X_stm = data['X'][idx, 0]
        X_afm = data['X'][idx, 1]
        Y = data['Y'][idx]
        xyz = unpad_xyz(np.array(data['xyz'][idx]))
    return X_stm, X_afm, Y, xyz 
    
def unpad_xyz(xyz):
    xyz = xyz[xyz[:, 3]>0]
    return xyz


if __name__ == '__main__':

    hdf5_path = 'dsh.hdf5' # Change this to whatever is the file name
    idx = 10613           # This is used as an index to get the data

    X_stm, X_afm, Y, xyz = get_data(hdf5_path, idx)

    print('Grid size:', X_stm.shape)      # (128, 128, 10) where 10 is the number of constant height images
    print('Number of atoms:', len(xyz))   # 41 for index 10613
    print('Descriptor shape:', Y.shape)   # (4, 128, 128) where 4 is the number of descriptors [Atomic Disks, vdW Spheres, HeightMap, ES Map]
    
    descriptors = ['ES Map', 'Atomic Disks', 'vdW Spheres', 'HeightMap']
    cmaps = ['bwr', 'viridis', 'viridis', 'viridis']
    
    fig = plt.figure(figsize=(14, 6))
    
    # ===== Plot scanning probe microscopy images ==========
    
    ax = plt.subplot(2, 5, 1)
    ax.imshow(X_stm[:, :, 0], cmap='gray', origin='lower')
    ax.set_title('STM far')

    ax = plt.subplot(2, 5, 2)
    ax.imshow(X_stm[:, :, -1], cmap='gray', origin='lower')
    ax.set_title('STM close')

    ax = plt.subplot(2, 5, 3)
    ax.imshow(X_afm[:, :, 0], cmap='afmhot', origin='lower')
    ax.set_title('AFM far')

    ax = plt.subplot(2, 5, 4)
    ax.imshow(X_afm[:, :, -1], cmap='afmhot', origin='lower')
    ax.set_title('AFM close')
    
    # ===== Plot molecule ===================================

    molecule = Atoms(positions=xyz[:, :3], numbers=xyz[:, 3])    
    ax = plt.subplot(2, 5, 5)
    plot_atoms(molecule, ax)
    ax.set_xlim(2, 18)
    ax.set_ylim(2, 18)
    ax.set_title('Top view')
    
    ax = plt.subplot(2, 5, 10)
    plot_atoms(molecule, ax, rotation=('-90x,0y,0z'))
    ax.set_xlim(2, 18)

    ax.set_title('Side view')
      
    # ===== Plot image descriptors ==========================
    
    es = Y[0]
    norm = mcolors.TwoSlopeNorm(vmin=es.min(), vmax = es.max(), vcenter=0)
    ax = plt.subplot(2, 5, 6)
    ax.imshow(es, cmap=cmaps[0], norm=norm, origin='lower')
    ax.set_title(descriptors[0])
    
    for i, y in enumerate(Y[1:]):
        ax = plt.subplot(2, 5, 7+i)
        ax.imshow(y, cmap=cmaps[i+1], origin='lower')
        ax.set_title(descriptors[i+1])
        
    plt.tight_layout()
    plt.show()
    
    # view(molecule) # Uncomment this if you want a full 3D visualization of the molecule
        
    
