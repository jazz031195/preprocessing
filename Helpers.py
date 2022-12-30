import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def print_volume(data: np.ndarray, time:int=15) -> None:
    """"
    Function that prints an overview of the z slices, at a fixed time=b-val
    Args:
        data (np.ndarray) : data to plot
        time (int) : time or b-value at which to plot the data 
    """

    nrows = 2
    ncols = int(data.shape[2]/nrows)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(10, 5))

    for i in range(data.shape[2]):
        if data.ndim == 4:
            slice = data[:, :, i,time]
        elif data.ndim == 3:
            slice = data[...,i]
        ax[i//ncols][i%ncols].imshow(slice.T, cmap="gray", origin="lower")

def save_nifti(input_img: np.ndarray, save_name: str, affine_transf:np.ndarray=np.eye(4), 
               header: nib.nifti1.Nifti1Header=[]) -> None:
    """
    Function that creates a Nifti1image and save it as .nii.gz
    Args:
        input_img (np.ndarray): array to save
        save_name (str): name to use to save
        affine_transf (np.ndarray): 4x4 array, affine transformation
        header (nib.nifti1.Nifti1Header): header of the Nifti file
    
    """
    if header:
        img = nib.Nifti1Image(input_img, affine_transf, header)
    else:
        img = nib.Nifti1Image(input_img, affine_transf)
        
    nib.save(img, save_name)

def load_affine(img_path: str) -> np.ndarray:
    """
    Function that returns the affine matrix from a nifti image
    Args:
        img_path (str): path of the nifti image
    Returns:
        affine matrix (np.ndarray)
    """
    
    img = nib.load(img_path)
    return img.affine