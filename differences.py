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

if parse(version('itk')) < parse('5.3'):
    raise ValueError("ITK greater than version 5.3.0 is required for this notebook")

#from itkwidgets import view
from matplotlib import pyplot as plt

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



def berechne_bild_differenz(bild1, bild2, string):
    
    
    # Sicherstellen, dass die Bilder dieselbe Größe haben
    if bild1.size != bild2.size:
        raise ValueError("Die Bilder müssen dieselbe Größe haben")
    
    """# In numpy Arrays umwandeln
    np_bild1 = np.array(bild1)
    np_bild2 = np.array(bild2)"""
    
    # Differenz berechnen (absolute Differenz zwischen den Pixelwerten)
    differenz = bild1 - bild2

    #print(differenz)
    #differenz = set_values_below_threshold_to_zero(differenz, 1)
    #print(differenz)
    bild2 = bild2 + differenz
    np.save('Differenz/bspline/'+string+ '.npy', differenz)
    #exit()
    fig, ax = plt.subplots()  # Create a figure and axis object
    ax.imshow(differenz, cmap='gray')  # 'gray' colormap for grayscale images

    # Remove the axis
    ax.axis('off')

    # Adjust the padding to remove the white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()
    
    plt.savefig('Differenz/bspline/'+string+ '.jpg')
    plt.close()
    
def set_values_below_threshold_to_zero(array, threshold):
    # Set entries smaller than the threshold to zero
    array[np.abs(array) < threshold] = 0
    return array


if __name__ == "__main__":
    plt.axis('off')
    #difference_reg_normal()
     
    #ent_list = load_sort_list('elastix_bspline_gridspacing')

    # fixed = 963
    fixed = np.load('elastix_bspline/952.npy')
    for k in range(951, 976, 1):
        print(k)
        
        moving = np.load('elastix_bspline/' + str(k) + '.npy')
        berechne_bild_differenz(fixed, moving, str(k)  )
        fixed = moving

    