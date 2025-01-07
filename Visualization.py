import matplotlib.pyplot as plt
import numpy as np
import re
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from os import listdir
from pathlib import Path

# Create four random images (for demonstration purposes)
image1 = np.random.rand(10, 10)
image2 = np.random.rand(10, 10)
image3 = np.random.rand(10, 10)
image4 = np.random.rand(10, 10)

def plot(image1, image2, image3, image4, k):
    # Create a figure and axis array with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot each image in its respective position
    axs[0, 0].imshow(image1, cmap='viridis')
    axs[0, 0].set_title("original")
    axs[0, 0].axis('off')  # Hide axes

    axs[0, 1].imshow(image2, cmap='viridis')
    axs[0, 1].set_title("Registration")
    axs[0, 1].axis('off')  # Hide axes

    axs[1, 0].imshow(image3, cmap='viridis')
    axs[1, 0].set_title("Raft")
    axs[1, 0].axis('off')  # Hide axes

    axs[1, 1].imshow(image4, cmap='viridis')
    axs[1, 1].set_title("Differenz")
    axs[1, 1].axis('off')  # Hide axes

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
    plt.savefig('Test_visualization/test_' + str(k) + '.jpg')

def plot_for_slide():
    string = 'elastix_bspline_gridspacing/'
    for k in range(1,23,1): #26
        print(k + 950)
        img1 = np.load(string + str(k + 950) + '.npy')
        img2 = np.load(string + str(k + 1 + 950) + '.npy')
        img3 = np.load(string + str(k + 2 + 950) + '.npy')
        img4 = np.load(string + str(k + 3 + 950) + '.npy')
        plot(img1,img2,img3,img4, k)

# Function to visualize the images
def Visualization(fixed_array, registered_array, moving, k):
    
        
    # Create a figure with 3 subplots (one for each image)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the fixed, registered, and moving images (all slices in the z-axis)
    axes[0].imshow(fixed_array, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title('Fixed Image Slice')
    
    axes[1].imshow(registered_array, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title('Registered Moving Image Slice')
    
    axes[2].imshow(moving, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title('Moving Image Slice')

    plt.tight_layout()
    plt.savefig('elastix_bspline_gridspacing/Visualization/' + str(k) +'.jpg')

def main():
  
    directory_path = 'results_slices/4dflow_KI_20241207-124205/71' # 951 - 975
   
    # get fixed - it's better to take an images in the middle, i.e. 951+12 = 963
    fixed = np.load(directory_path + '/963.npy') 
    
    for k in range(1,26,1): #26
        
        moving = directory_path + '/' + str(k + 950) + '.npy'
        path_reg = 'elastix_bspline_gridspacing/'
        path_raft = 'Raft/elastix_bspline_gridspacing/'
        path_diff = 'Differenz/bspline/'
        #print(string)
        
        movarr_original = np.load(moving)

        # chech if registrated images exists and load otherwise continue
        my_file = Path(path_reg + str(k + 950) + '.npy')
        if my_file.is_file():
            reg = np.load(path_reg + str(k + 950) + '.npy')
        else:
            print(str(my_file) + ' does not exist')
            
            continue


        if k == 1:
            img_raft_string = path_raft + str(k + 1 + 950) + '.png'
        else:
            img_raft_string = path_raft + str(k + 950) + '.png'

        raft_img = plt.imread(img_raft_string)

        img_diff_path = path_diff + str(k + 950) + '.jpg'
        img_diff = plt.imread(img_diff_path)

        #plot(movarr_original, reg, raft_img, img_diff, k)
        Visualization(fixed, reg, movarr_original, k + 950)
        plt.close()

if __name__ == "__main__":

  
    print('main:')
    main()
    
    
    print('end of program')
    exit()

    #plot(image1, image2, image3, image4, 99)
    #plot_for_slide()

