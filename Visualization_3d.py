import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt

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
def Visualization(fixed_array, registered_array, moving, output_folder):
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
        plt.savefig( output_folder + '/' + str(k) +'.jpg')

# Function to load and visualize 3D images
def load_3d_and_visualize(path_to_registrated_image, time_fixed, time_moving, output_folder):
    img_path = path_to_registrated_image
    reg_image = np.load(img_path)
    
    # Load fixed and moving 3D images
    image_fixed = get3d_48_62(time_fixed)
    image_moving = get3d_48_62(time_moving)
    image_fixed_array = sitk.GetArrayFromImage(image_fixed)
    image_moving = sitk.GetArrayFromImage(image_moving)
    # Visualize fixed and registered images
    Visualization(image_fixed_array, reg_image, image_moving, output_folder)

if __name__ == "__main__":
    path_to_registrated_image = 'elastix_3d/7_to_12.npy'
    load_3d_and_visualize(path_to_registrated_image, 12, 7,'elastix_3d/Visu_7_to_12')  # Visualize fixed and registered images

    print("End of program")