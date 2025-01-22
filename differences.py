# calculating differences
from PIL import Image
import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itk
from configparser import ConfigParser
import numpy as np
#import itkJPEGImageIO
#include <itkJPEGImageIO.h>
import itk
from os import listdir
from packaging.version import parse
from importlib.metadata import version
import torch
import torchvision.transforms.functional as F
from pathlib import Path
from split_3d import split_3d_to_2d

if parse(version('itk')) < parse('5.3'):
    raise ValueError("ITK greater than version 5.3.0 is required for this notebook")

#from itkwidgets import view


#matplotlib inline
#from ipywidgets import interact


path_new = 'out_4d_jpg/new_20241114-153739/56_new'
path_diff = 'diff_images/registration_img'

path_reg_normal = 'reg_normal'
path_slice_18 = 'Slice_18'
path__ohne_reg = 'resNpSlices/4dflow_KI_20241104-233450/71'
raft_tensor = 'raft_reg_tensor'

def difference_tensor():
    np.set_printoptions(threshold=sys.maxsize)
    entity_list = []
    for image in listdir(raft_tensor):
        print(image)
        imagestring = image.removesuffix('.pt')
        print(type(image))
        print(imagestring)
        #intimagestring = int(imagestring)
        moving = torch.load(raft_tensor +'/' + image)
        
        print(moving.shape)

        entity = [int(imagestring), moving]
        entity_list.append(entity)

    entity_list.sort() 
    print('sorted')
    #print(entity_list[0][0])
    #print(entity_list[0][1])

    first = True
    for ent in entity_list:
        print(ent[0])
        if (first):
            sum = ent[1]
            first = False
            continue
        print('Do summation')
        sum = sum + ent[1]
    

    print('stuff')
    print(sum.shape)
    print(type(sum))
    print('stuff ended')
    print(sum)
    sum = torch.div(sum, 25)
    print('devided')
    print(sum)

    first = True
    for ent in entity_list:
        #print('ent[1]')
        #print(ent[1].max())
        #print(ent[1].min())
        diff = entity_list[12][1] - ent[1]
        print('diff')
        #print(diff.max())
        #print(diff.min())
        if (first):
            print(diff)
            first = False
            
        print(type(diff))
        print(diff.shape)
        print(diff)
        #np.save('diff_images/differences/img_npy'+'/'+str(ent[0]), diff)
        #diff.permute(1,2,0)
        img = F.to_pil_image(diff.to("cpu"))
        plt.imshow(img)
        plt.savefig('tensor_diff_raft/notmean'+'/'+'pilimage.jpg')
        diff = diff.cpu()
        diff = diff.numpy()
        print(type(diff))
        print(diff.shape)
        #print(diff)
        diff = np.transpose(diff, (1,2,0))
        print(diff.shape)
        plt.imshow(diff, cmap='gray')
        plt.savefig('tensor_diff_raft/notmean'+'/'+str(ent[0])+'.jpg')

        diff = sum - ent[1]
        print('mean')
        diff = diff.cpu()
        diff = diff.numpy()
        print(type(diff))
        print(diff.shape)
        diff = np.transpose(diff, (1,2,0))
        print(diff.shape)
        #print(diff.max())
        #print(diff.min())
        #np.save('diff_images/mean_differences/img_npy'+'/'+str(ent[0]), diff)
        plt.imshow(diff, cmap='gray')
        plt.savefig('tensor_diff_raft/mean'+'/'+str(ent[0])+'.jpg')

        exit()

def difference_reg_normal():
    entity_list = []
    for image in listdir(path_reg_normal):
        print(image)
        #imagestring = image.removesuffix('.npy')
        if image.endswith('.png'):
            #print('png')
            continue
        else:
            imagestring = image.removesuffix('.npy')
       
        
        print(imagestring)
        intimagestring = int(imagestring)
        moving = np.load(path_reg_normal +'/' + image)
        print(moving.shape)
        #plt.imshow(moving, cmap='gray')
        #plt.savefig('diff_images/differences/no_diff'+'/'+imagestring+'.jpg')

        entity = [int(imagestring), moving]
        entity_list.append(entity)

    entity_list.sort() 
    print('sorted')
    #print(entity_list[0][0])
    #print(entity_list[0][1])

    first = True
    for ent in entity_list:
        print(ent[0])
        if (first):
            sum = ent[1]
            first = False
            continue
        #print('Do summation')
        sum = sum + ent[1]
    

    """print('stuff')
    print(sum.shape)
    print(type(sum))
    print('stuff ended')
    #print(sum)"""
    sum = np.divide(sum, 25)
    print('devided')
    #print(sum)  

    first = True
    for ent in entity_list:
        #print('ent[1]')
        #print(ent[1].max())
        #print(ent[1].min())
        diff = entity_list[0][1] - ent[1]
        print('diff')
        #print(diff.max())
        #print(diff.min())
        if (first):
            print(diff)
            first = False
            
        #print(type(diff))
        #print(diff.shape)
        np.save('diff_images/differences/reg_normal'+'/'+str(ent[0]), diff)
        plt.imshow(diff, cmap='gray')
        plt.savefig('diff_images/differences/reg_normal'+'/'+str(ent[0])+'.jpg')

        diff = ent[1] - sum
        print('mean')
        #print(diff.max())
        #print(diff.min())
        np.save('diff_images/differences/reg_normal_mean'+'/'+str(ent[0]), diff)
        plt.imshow(diff, cmap='gray')
        plt.savefig('diff_images/differences/reg_normal_mean'+'/'+str(ent[0])+'.jpg')


def load_sort_list(path):
    entity_list = []
    for image in listdir(path):
        print(image)
        #imagestring = image.removesuffix('.npy')
        if image.endswith('.jpg'):
            #print('png')
            continue
        else:
            imagestring = image.removesuffix('.npy')
       
        
        print(imagestring)
        #intimagestring = int(imagestring)
        moving = np.load(path +'/' + image)
        print(moving.shape)
        #plt.imshow(moving, cmap='gray')
        #plt.savefig('diff_images/differences/no_diff'+'/'+imagestring+'.jpg')

        entity = [int(imagestring), moving]
        entity_list.append(entity)

    entity_list.sort() 
    print('sorted')
    return entity_list

def print_highest_and_lowest_values(arr):
    # Ensure the array is a NumPy array
    arr = np.asarray(arr)
    
    # Flatten the 3D array into a 1D array
    flattened_arr = arr.flatten()
    
    # Sort the flattened array
    sorted_arr = np.sort(flattened_arr)
    
    # Get the 10 smallest values (first 10 elements)
    lowest_values = sorted_arr[:10]
    
    # Get the 10 largest values (last 10 elements)
    highest_values = sorted_arr[-10:]
    
    # Print the results
    print("10 Lowest Values:")
    print(lowest_values)
    
    print("\n10 Highest Values:")
    print(highest_values)

def berechne_bild_differenz(bild1, bild2, string):
    
    
    # Sicherstellen, dass die Bilder dieselbe Größe haben
    if bild1.size != bild2.size:
        raise ValueError("Die Bilder müssen dieselbe Größe haben")
    
    """# In numpy Arrays umwandeln
    np_bild1 = np.array(bild1)
    np_bild2 = np.array(bild2)"""
    
    # Differenz berechnen (absolute Differenz zwischen den Pixelwerten)
    #print('bild 1')
    #print_highest_and_lowest_values(bild1)
    differenz = bild1 - bild2
    #print(np.min(differenz))
    #differenz = differenz + abs(np.min(differenz))
    #print('differenz')
    #print_highest_and_lowest_values(differenz)
    #print(differenz)
    #differenz = set_values_below_threshold_to_zero(differenz, 1)
    #print(differenz)
   # bild2 = bild2 + differenz
    np.save('Differenz/elastix_bspline_gridspacing/'+string+ '.npy', differenz)
    #exit()
    fig, ax = plt.subplots()  # Create a figure and axis object
    ax.imshow(differenz, cmap='gray')  # 'gray' colormap for grayscale images

    # Remove the axis
    ax.axis('off')

    # Adjust the padding to remove the white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
    
    plt.savefig('Differenz/elastix_bspline_gridspacing/'+string+ '.jpg')
    plt.close()
    
def set_values_below_threshold_to_zero(array, threshold):
    # Set entries smaller than the threshold to zero
    array[np.abs(array) < threshold] = 0
    return array


# Function to load 3D data from .npy files
def get3d_48_62(timeslice):
    if timeslice < 0 or timeslice > 24:
        raise IndexError('slices only between 0 and 25')
    
    vectorOfSubImages = sitk.VectorOfImage()
    path = 'results_slices/4dflow_KI_20241207-124205/'

    for k in range(48, 63):  # Slice numbers between 48 and 62
        slice_num = 109 - k
        slicenumber = slice_num * 25 + timeslice
        k_path = os.path.join(path, str(k), f'{slicenumber}.npy')
        
        # Load .npy file and convert to SimpleITK image
        try:
            arr_loaded = np.load(k_path)
            arr_img = sitk.GetImageFromArray(np.float32(arr_loaded))
            vectorOfSubImages.push_back(arr_img)
        except FileNotFoundError:
            print(f"Warning: {k_path} not found. Skipping slice.")
    
    return sitk.JoinSeries(vectorOfSubImages)


# Function to visualize the images
def Visualization(fixed_array, registered_array, moving):
    for k in range (0,15,1):
        print(k)
    
    
        slice_idx = k  # You can change this slice index
        
        # Create a figure with 3 subplots (one for each image)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot the fixed, registered, and moving images (all slices in the z-axis)
        axes[0].imshow(fixed_array[slice_idx, :, :], cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Fixed Image Slice')
        
        axes[1].imshow(registered_array[slice_idx, :, :], cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Registered Moving Image Slice')
        
        axes[2].imshow(moving[slice_idx, :, :], cmap='gray')
        axes[2].axis('off')
        axes[2].set_title('Moving Image Slice')

        plt.tight_layout()
        plt.savefig('elastix_3d/Visu_7_to_12/' + str(k) +'.jpg')


    

def compute_slicewise_difference(image1, image2):
    # Load the 3D images from the .npy files
    """image1 = np.load(image1_path)
    image2 = np.load(image2_path)"""
    
    # Ensure the two images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same shape.")
    
    # Get the number of slices (along the z-dimension)
    z_dim = image1.shape[0]
    
    # Compute the difference slice by slice along the z-dimension
    differences = np.zeros_like(image1)
    
    for z in range(z_dim):
        differences[z] = image1[z] - image2[z]
    
    return differences

def Differenz_3d():
    """ path_slice = 'elastix_3d/slices_7_to_12/slice_0'
    fixed = np.load('elastix_3d/slices_7_to_12/slice_001.npy')
    fixed_path = 
    moving_path = 'elastix_3d/7_to_12.npy'"""

    img_path = 'elastix_3d/7_to_12.npy'
    reg_image = np.load(img_path)
    
    # Load fixed image at a specific time
    time_fixed = 12
    time_moving = 7
    
    # Load fixed and moving 3D images
    image_fixed = get3d_48_62(time_fixed)
    image_moving = get3d_48_62(time_moving)
    image_fixed_array = sitk.GetArrayFromImage(image_fixed)
    image_moving = sitk.GetArrayFromImage(image_moving)

    differences = compute_slicewise_difference(image_fixed_array, reg_image)
    np.save('Differenz/3d_7_to_12/Differences_7_to_12.npy', differences)
    
    # split 3d differences
    input_path = 'Differenz/3d_7_to_12/Differences_7_to_12.npy'  # Path to the input 3D image
    output_dir = 'Differenz/3d_7_to_12'  # Folder to save the 2D slices

    # Split the 3D image into 2D slices and save them
    split_3d_to_2d(input_path, output_dir)

    # Visualize fixed and registered images
    #Visualization(image_fixed_array, reg_image, image_moving)

    """for k in range(0, 15, 1):
        #print(k)
        if k < 10 :
            path_image = path_slice +str(0)+str(k)+'.npy'
        else:
            path_image = path_slice +str(k)+'.npy'
        
        moving = np.load(path_image)

        Differenz = fixed - moving
        #np.save('Differenz/3d_7_to_12' +string+ '.npy', differenz)
        #exit()
        fig, ax = plt.subplots()  # Create a figure and axis object
        ax.imshow(differenz, cmap='gray')  # 'gray' colormap for grayscale images

        # Remove the axis
        ax.axis('off')

        # Adjust the padding to remove the white border
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.show()
        
        plt.savefig('Differenz/elastix_bspline_gridspacing/'+string+ '.jpg')
        plt.close()
        #print(path_image)
        fixed = moving"""


def Differenz_2d():
    # fixed = 963
    path_image = 'elastix_bspline_gridspacing/952.npy'

    my_file = Path(path_image)
    if my_file.is_file():
        fixed = np.load('elastix_bspline_gridspacing/952.npy')
    else:
        print('fixed path does not exist -> exit')

        exit()

    for k in range(951, 976, 1):
        print(k)
        path_image = 'elastix_bspline_gridspacing/' + str(k) + '.npy'
        my_file = Path(path_image)
        if my_file.is_file():
             moving = np.load('elastix_bspline_gridspacing/' + str(k) + '.npy')
        else:
            print('moving path does not exist -> continue')
            continue

        berechne_bild_differenz(fixed, moving, str(k)  )
        fixed = moving

if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    plt.axis('off')
    #difference_reg_normal()
     
    #ent_list = load_sort_list('elastix_bspline_gridspacing')

   
    #Differenz_2d()
    
    Differenz_3d()