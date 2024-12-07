# calculating differences

import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itk
from configparser import ConfigParser
import numpy as np
#xximport itkJPEGImageIO
#include <itkJPEGImageIO.h>
import itk
from os import listdir
from packaging.version import parse
from importlib.metadata import version
import torch
import torchvision.transforms.functional as F

if parse(version('itk')) < parse('5.3'):
    raise ValueError("ITK greater than version 5.3.0 is required for this notebook")

import numpy as np
#from itkwidgets import view
from matplotlib import pyplot as plt

#matplotlib inline
from ipywidgets import interact


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


def sort_list(path):
    entity_list = []
    for image in listdir(path):
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
    return entity_list

def plot_side_by_side(path_1, path_2):
    for number in range(952,976,1):
        print(number)
        f, axarr = plt.subplots(1,2)
        img1 = mpimg.imread('movie_folder/raft_reg/' + str(number) + '.png')
        img2 = mpimg.imread('movie_folder/nonreg_raft/' + str(number) + '.png')
        #imgplot = plt.imshow(img)
        #plt.show()
        axarr[0].imshow(img1)
        axarr[1].imshow(img2)
        plt.savefig('movie_raft/vergleich/' + str(number) + '.png')

if __name__ == "__main__":
     #difference_reg_normal()
     plot_side_by_side('movie_folder/nonreg_raft', 'movie_folder/raft_reg')