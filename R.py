# Sa/So:

# visualisiere torch

# do differenzes
# training of raft?

# Do reg vor all slices
# do reg for 15 slices
# do differenz and then reg
# do reg and then differenz (standard)
# do raft on differenzes
# do differenzes on raft (I did, makes no sense right, since Raft does the flow and differences also)


# make movies
# reg 3d
# original
# reg normal done 
# differences reg done
# diff reg mean done
# diff Raft reg done
# diff Raft reg mean done
# Raft reg done 
# Send alexander a mail




#import SimpleITK as sitk
from os import listdir
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
import matplotlib.image as mpimg

import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large

from torchvision.utils import flow_to_image
import sys


plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
#plt.rcParams["savefig.frameon"] = "none"
# sphinx_gallery_thumbnail_number = 2


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            plt.imshow(img)
            plt.savefig('img_rgb')
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(368, 368)),
        ]
    )
    batch = transforms(batch)
    return batch




print('warning-------------------------')
def main_2(t_1,t_2,counter):
    t_1 = torch.from_numpy(t_1)
    t_2 = torch.from_numpy(t_2)
    t_1.unsqueeze_(0)
    t_2.unsqueeze_(0)
    print(t_1.shape)
    t_1 = t_1.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    t_2 = t_2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
    print(t_1.shape)
    img_batchlist_1 = t_1
    img_batchlist_2 = t_2
    print(f"shape = {img_batchlist_1.shape}, dtype = {img_batchlist_1.dtype}")
    print(f"shape = {img_batchlist_2.shape}, dtype = {img_batchlist_2.dtype}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device = ')
    print(device)
    
    img_batchlist_1 = preprocess(img_batchlist_1).to(device)
    img_batchlist_2 = preprocess(img_batchlist_2).to(device)

    print(f"shape = {img_batchlist_1.shape}, dtype = {img_batchlist_1.dtype}")
    print(f"shape = {img_batchlist_2.shape}, dtype = {img_batchlist_2.dtype}")
    plot(img_batchlist_1)
    plt.savefig("R_img1_batch")
    
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()

    print('warning-------------------------')

# list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    list_of_flows_2 = model(img_batchlist_1.to(device), img_batchlist_2.to(device))

    print(f"type = {type(list_of_flows_2)}")
    print(f"length = {len(list_of_flows_2)} = number of iterations of the model")

    predicted_flows_2 = list_of_flows_2[-1]
    print(f"dtype = {predicted_flows_2.dtype}")
    print(f"shape = {predicted_flows_2.shape} = (N, 2, H, W)")
    print(f"min = {predicted_flows_2.min()}, max = {predicted_flows_2.max()}")

    flow_imgs_2 = flow_to_image(predicted_flows_2)
    print(type(flow_imgs_2))
    for flow_img in flow_imgs_2:
        print(type(flow_img))

    img_batchlist_1 = [(img1 + 1) / 2 for img1 in img_batchlist_1]

    for (img1, flow_img) in zip(img_batchlist_1, flow_imgs_2):
        print(img1.shape)
        flow_img = flow_img.cpu()
        flow_img = flow_img.numpy()
        flow_img = np.transpose(flow_img, (1,2,0))
        plt.imshow(flow_img)
        plt.savefig("raft_reg/" + str(counter))

        #torch.save(flow_img, "raft_reg_tensor/" + str(counter) +".pt")
        """grid = [img1, flow_img] 
        plot(grid)
        plt.savefig("raft_diff_reg_normal_mean/" + str(counter))"""
        
        
        



####################################
# Bonus: Creating GIFs of predicted flows
# ---------------------------------------
# In the example above we have only shown the predicted flows of 2 pairs of
# frames. A fun way to apply the Optical Flow models is to run the model on an
# entire video, and create a new video from all the predicted flows. Below is a
# snippet that can get you started with this. We comment out the code, because
# this example is being rendered on a machine without a GPU, and it would take
# too long to run it.

from torchvision.io import write_jpeg
"""for i, (img1, img2) in enumerate(zip(frames, frames[1:])):

    print(i)
    print(img1)"""
    
"""     # Note: it would be faster to predict batches of flows instead of individual flows
     img1 = preprocess(img1[None]).to(device)
     img2 = preprocess(img2[None]).to(device)

     list_of_flows = model(img1_batch, img2_batch)
     predicted_flow = list_of_flows[-1][0]
     flow_img = flow_to_image(predicted_flow).to("cpu")
     output_folder = "output"  # Update this to the folder of your choice
     write_jpeg(flow_img, output_folder + f"predicted_flow_{i}.jpg")"""

####################################
# Once the .jpg flow images are saved, you can convert them into a video or a
# GIF using ffmpeg with e.g.:
#
# ffmpeg -f image2 -framerate 30 -i predicted_flow_%d.jpg -loop -1 flow.gif

def load_reg_normal():

    arr_list = []
    name_list = []
    unsorted_list = []
    image_npy_list = []
    directory_path = 'reg_normal'
    return_list = []
    #name_list.append('reg')
    for dir_content in listdir(directory_path):
        print(dir_content)
        if dir_content.endswith('png'):
            #print('png')
            continue
        else:
            
            test = 'test vor variables'
            movarr_loaded = np.load(directory_path +'/' + dir_content)
            # movarr_loaded = np.float32(movarr_loaded)
            #moving = sitk.GetImageFromArray(movarr_loaded)
            arr_list.append(movarr_loaded)
            string = dir_content
            string_2 = string.removesuffix('.npy')
            name_list.append(string_2)
            #image_npy_list.append(moving)
            print(dir_content)
            number = int(string_2)
            #print(number)
            #print(type(number))
            #print(string_2)
            #print(type(string_2))
            entity = [number, string_2, movarr_loaded]
            unsorted_list.append(entity)

    print(name_list)
    # print(unsorted_list)
    sorted_list = sorted(unsorted_list)
    print('sorted list -------------')
    #print(sorted_list)


    return sorted_list


"""def load_data():


    arr_list = []
    name_list = []
    unsorted_list = []
    image_npy_list = []
    directory_path = '71'
    return_list = []
    #name_list.append('reg')
    for dir_content in listdir(directory_path):
        test = 'test vor variables'
        movarr_loaded = np.load(directory_path +'/' + dir_content)
       # movarr_loaded = np.float32(movarr_loaded)
        #moving = sitk.GetImageFromArray(movarr_loaded)
        arr_list.append(movarr_loaded)
        string = dir_content
        string_2 = string.removesuffix('.npy')
        name_list.append(string_2)
        #image_npy_list.append(moving)
        #print(dir_content)
        number = int(string_2)
        #print(number)
        #print(type(number))
        #print(string_2)
        #print(type(string_2))
        entity = [number, string_2, movarr_loaded]
        unsorted_list.append(entity)
    
    print(name_list)
   # print(unsorted_list)
    sorted_list = sorted(unsorted_list)
    print('sorted list -------------')
    #print(sorted_list)
    

    return sorted_list"""

"""def tensor_production(grey_image):
    torch.set_printoptions(profile="full")
   
    
    
    print(grey_image.shape)
    plt.imshow(grey_image) 
    #plt.figure(frameon=False)
    plt.savefig('tensorproduction',edgecolor='red') 
    img=mpimg.imread('tensorproduction.png')
    print(img.shape)
    tensor_images = torch.tensor(img)
    #tensor_images[380:9,:,:] = 0
    print(tensor_images.shape)
    tensor_images = torch.tensor(tensor_images[10:])
    tensor_images = torch.tensor(tensor_images[:,10:])
    tensor_images = torch.tensor(tensor_images[:370])
    tensor_images = torch.tensor(tensor_images[:,:370])
    print(tensor_images.shape)

    tensor_images = torch.tensor(tensor_images[:,:, : 3])
    print(tensor_images.shape)
    plt.imshow(tensor_images)
    plt.savefig('tensor')
    torch.set_printoptions(profile="default") # reset
    return tensor_images"""


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    torch.set_printoptions(precision=True)
    plt.margins(x=0,y=0)
    #gca().spines[['right', 'top']].set_visible(False)
    plt.axis('off')  # Hide axes 
    plt.box(on=None)
    print('load_data:')
    #data = load_data()
    data = load_reg_normal()
    
    print(len(data))
    for counter in range(0,len(data)-1,1):
      
        
        print('pictures processed: ' + str(data[counter][0]) + ' and ' + str(data[counter + 1][0]))
        #print(data[counter + 1][0])
       # import SimpleITK as sitk
        #fixed = sitk.GetImageFromArray(data[counter][2])
        #print(fixed.shape)
        #print(fixed.ty)
        img1 = data[counter][2]
        img2 = data[counter+1][2]
        print('length')
        print(len(img1.shape))
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        print(img1.shape)
        """if (counter == 2):
            rgba_1 = tensor_production(grey_1)
            rgba_2 = tensor_production(grey_2)
            rgba_1.unsqueeze_(0)
            rgba_2.unsqueeze_(0)
            print(rgba_1.shape)
            #rgba_1[0] = 1
            print(rgba_1.shape)"""

        print('main_2:')
        main_2(img1, img2, data[counter + 1][0])

        #print(rgba_1.dtype)
        #print(rgba_1.shape)
        """t_1 = torch.from_numpy(data[counter][2])
        print('data --------')
        print(data[counter][2].shape)
        print(data[counter][2].dtype)
        t_2 = torch.from_numpy(data[counter+1][2])
        print(t_1.shape)
        print(t_1.dtype)
        if(counter == (len(data)-3)):
            
            plt.imshow(t_1)
            plt.savefig('img1_gray')
        t_1.unsqueeze_(0)
        t_1 = t_1.repeat(3, 1, 1)
        t_1 = t_1.permute(1, 2, 0)
        if(counter == (len(data)-3)):
            print(t_1.shape)
           # t_1 = t_1.permute(1, 2, 0)
            print(t_1.shape)
            plt.imshow(t_1)
            plt.savefig('img1_rgb')
        t_1.unsqueeze_(0)
        t_1[0] = 1
        t_2.unsqueeze_(0)
        t_2 = t_2.repeat(3, 1, 1)
        t_2 = t_2.permute(1, 2, 0)
        t_2.unsqueeze_(0)
        t_2[0] = 1"""
        if(counter >= (len(data)-2)):
            """print('t1 --------')
            #print(t_1)
            
            
            print(t_1.shape)
            print(t_1.dtype)
            print('t2 --------')
            #print(t_1)
            
            
            print(t_2.shape)
            print(t_2.dtype)"""
            print('number of run: ' + str(counter))
            # main_2(t_1, t_2, data[counter + 1][0])
        
