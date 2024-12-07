from configparser import ConfigParser
from os import listdir
import time
import itk
import sys
import os.path as osp
import os
import matplotlib.pyplot as plt
import pydicom
from pydicom.fileset import FileSet
from pydicom.filereader import read_dicomdir
from pydicom.data import get_testdata_files
from pydicom.data import get_testdata_file
from pydicom import dcmread
from os.path import dirname, join
from pprint import pprint
from pydicom.filereader import read_dicomdir
from os.path import dirname, join
from pprint import pprint
from PIL import Image
import glob
#import utilities.path_utils as path_utils
import itk
import SimpleITK as sitk
import numpy as np
#from itk_test import ComputeMetrik
#from Registration_test import testmain_2

ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '../../'))
sys.path.append(ROOT_DIR)

def create_save_dir(OUTPUT_PATH, name):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    saveDir = os.path.sep.join([OUTPUT_PATH, name + "_" + timestr])
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    # print("save results to directory: ", saveDir, "\n")
    return saveDir

"""def writing(data):
    headline = 0
    with open("Slice_18.txt", "w") as txt_file:
        for number in data:
           
           txt_file.write(str(headline)  + "\n")
           headline += 1
           print('test')
           array_test = ds.pixel_array
           print(array_test.shape)
           print(array_test.dtype)
           #print(number.dtype)
           txt_file.write(" ".join(str(array_test)) ) # + "\n")"""

def create_sub_dir(saveDir, SUBDIR_PATH):
    subDir = os.path.sep.join([saveDir, SUBDIR_PATH])
    if not os.path.exists(subDir):
        os.makedirs(subDir)
    return subDir

def save_config(config, save_dir, filename='config.ini'):
    # save config file to save directory
    fout = osp.join(save_dir, filename)
    with open(fout, 'w') as config_file:
        config.write(config_file)

if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read('parser/itk.ini')
    data = cfg['DATA']
    vizcfg = cfg['VIZ']

    # Source paths
    root_dir = data['root_dir']
    print(root_dir)

    # Output paths
    save_dir = create_save_dir(data['output_dir'], f'{osp.basename(root_dir)}')
    save_config(cfg, save_dir)
    #imgs_dir = create_sub_dir(save_dir, 'test') # creates subdir images

path = '4dflow_KI/4dflow_Dicom/4dflow_KI_neu_bivent/DICOMDIR'

# fetch the path to the test data
print('Path to the DICOM directory: {}'.format(path))
# load the data
ds2 = dcmread(path)
fs = FileSet(ds2)  # or FileSet(path)
#print(fs)
#print(fs.path)
"""
print("showing instances")
counter = 0
for e in ds2:
    counter += 1
    print(e)
    print('counter_3 = ' , counter)
    if counter == 5:
        counter_2 = 0
        for instance in e:
            #print(instance)
            #print('next instance: ' + str(counter_2))
            #if (counter_2 % 100 == 0):
                #print(counter_2)    
                #node = instance.node
                #print(node.record_type)
            counter_2 +=1    
        print(counter_2)        
"""
testcounter = 0
for slicenumber in range(0,110,1):
    slice = str(slicenumber)
    slice_dir = create_sub_dir(save_dir, slice) # creates subdir images
    




# save pictures in output directory
Slice = 8 #example: 72 = 19.8126259622197    --- not relevant, the code will do all slices now
list_number = Slice + 53
print('Slice = ' +str(Slice))
print((str(Slice) + '.8126259622197'))
img_counter = 0
slicecounter = 0
serie_number = 0
save_counter = 0
serie_counter = 0
counter_fs = 0
SliceLocation_list = []
image_list_slice = []
image_list_slice_names = []
counter_new_serie = 0
ListQ = True

#slice_dir = path_utils.create_sub_dir(imgs_dir, data['slice_folder_' + str(Slice)]) # creates subdir Serie
for instance in fs:
    ds = instance.load() # loads different iamges /whatever is in fs i.e. rawdata
    Instance_ser_num = ds.SeriesNumber
    if(Instance_ser_num != serie_number):
        serie_number = Instance_ser_num
        print('Serie has ' + str(counter_new_serie) +' images')
        counter_new_serie = 0
        print('New Serie: ' + str(Instance_ser_num))
        #serie_number = Instance_ser_num
        #serie_dir = path_utils.create_sub_dir(imgs_dir, data['serie_folder_' + str(serie_number)]) # creates subdir Serie       
        node = instance.node
    node = instance.node     
    if(node.record_type == 'RAW DATA'):
        print('RAW DATA')
    else:
        if(ds.SliceLocation not in SliceLocation_list):
            SliceLocation_list.append(ds.SliceLocation)
            #print('appended')
    if(Instance_ser_num == 2503): #110 * 25 = 2750 images + 1 raw data / magnituden serie
        if ListQ:
            SliceLocation_list.sort()
            print('sorted!')
            ListQ = False
            print(SliceLocation_list)
            print(len(SliceLocation_list))
            NpLocList = np.array(SliceLocation_list)
            print(NpLocList)
        #print(len(SliceLocation_list))
        #print(SliceLocation_list)
        #print('slicecounter = ' + str(slicecounter) )
        #print(SliceLocation_list[slicecounter])
        #string_slice = (str(Slice) + '.8126259622197')
        #print(string_slice)
        
        if(node.record_type == 'RAW DATA'):
            print('RAW DATA')
        else:
            if (np.where(NpLocList == ds.SliceLocation)[0][0] < 1):
                print(np.where(NpLocList == ds.SliceLocation)[0][0])
                print(ds.SliceLocation)
            if (np.where(NpLocList == ds.SliceLocation)[0][0] > 108):
                print(np.where(NpLocList == ds.SliceLocation)[0][0])
                print(ds.SliceLocation)
            string = str(instance.InstanceNumber)
            slice = str(np.where(NpLocList == ds.SliceLocation)[0][0])
            path = str(save_dir)
            np.save(path + '/'+ slice +'/'+ string+'.npy',ds.pixel_array)
            if(testcounter < 10):
                print(path + '/'+ slice +'/'+ string+'.npy')
                testcounter += 1

        
        
        
            
            
    counter_new_serie +=1
    counter_fs += 1

print('end of program 1')




# Old code - Just saved here in case I need old ideas




"""if(node.record_type == 'RAW DATA'):
            print('RAW DATA')
        elif (ds.SliceLocation == SliceLocation_list[list_number]):
            slicecounter +=1
            image_list_slice.append(ds)
            string = str(instance.InstanceNumber)
            image_list_slice_names.append(string)"""


#print("SliceLocation: " + str(ds.SliceLocation))
            #print('counter_ new_serie  ' + str(counter_new_serie))
"""
            if (counter_new_serie % 100 == 0 ):
                print('counter_ new_serie % 100 ' + str(counter_new_serie))
                print('New directory will be opened')
                serie_counter += 1
                if(serie_counter == 27):
                    serie_dir = path_utils.create_sub_dir(imgs_dir, data['serie_folder_' + str(serie_counter)]) # creates subdir Serie
            if(serie_counter == 27):
            """
                #if(img_counter < 3):
                #print('ds:')
            #print(ds.pixel_array)
            
            
            #image = itk.imread(ds.pixel_array)
            #image = sitk.Image(ds.pixel_array)
                #print(ds.PatientName)
                #print(ds.PatientSex)
                #(print('instance:'))
                #print(instance)
                #print(ds.pixel_array)
                #print(ds.pixel_array.shape)
                #print("SliceLocation: " + str(ds.SliceLocation))
                #print("StudyTime: " + str(ds.StudyTime))
                #img_counter += 1
            #im = plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
                #plt.savefig('foo1.pdf')
            #string = str(instance.InstanceNumber) + '.jpg'
            #print(string)
                #if(string == '2733.jpg'):
                    #break
            #plt.savefig(slice_dir + '/' + string) 
                #counter += 1
                #ims2.append([plt])
                #print("counter = " , counter)  
                #print('Image')
            #fixed_image = itk.imread(ds)
            #print(fixed_image)
            #file_reader = itk.ImageFileReader(instance)
            #image = itk.ReadImage(instance) 
            #img_arr = itk.GetArrayFromImage(image[:,:,0]) 
            ##plt.imshow(img_arr)
            #input_image = itk.imread(ds.pixel_array)
            #print(input_image)
            #if(slicecounter > 4):
                #break

#if (counter_fs % 100 == 0):
            
            #print(counter_fs)    
            #print(instance.path)
            #print(node.record_type)
            
 
  

"""print('counter_fs = ' + str(counter_fs))
print('Slice ' + str(Slice) +' has ' + str(slicecounter) +" images")
#print(len(SliceLocation_list))

print(len(image_list_slice))

c_for = 0
print('image_list_slice:')"""
"""for ds in image_list_slice:
    np.save('slice_'+ str(Slice) +'_npy/'+image_list_slice_names[c_for]+'.npy',ds.pixel_array)
  
    print(c_for)
    
    array_test = ds.pixel_array
    print(array_test.shape)
    print(array_test.dtype)
    #print(array_test.data)
    #print(array_test)   
    print(image_list_slice_names[c_for])
    c_for += 1"""
#Writing slice list in a file
#writing(image_list_slice)
"""

        itk_test_1 = itk.image_view_from_array(array_test)
        print(f"ITK image data pixel value at [2,1] = {itk_test_1.GetPixel([2,1])}")
        print(itk.size(itk_test_1))
        #print(f"ITK image data pixel value at [2,1] = {itk_test_1.GetPixel()}")
        #array_np_test = pydicom.pixel_data_handlers.numpy_handler.get_pixeldata(ds)
        #print(array_np_test)
        #print(array_np_test.shape)
        #print(array_np_test.dtype)
        #itk_test_2 = itk.image_view_from_array(array_np_test)
        np_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.int32)
        itk_np_view_data = itk.image_view_from_array(np_data)

        print(f"ITK image data pixel value at [2,1] = {itk_np_view_data.GetPixel([2,1])}")
        print(f"NumPy array pixel value at [2,1] = {np_data[2,1]}")
"""


#testmain_2(image_list_slice[0].pixel_array,image_list_slice[3].pixel_array)
#fixed_image_2 = itk.image_view_from_array(image_list_slice[0].pixel_array)
#print(fixed_image_2.dtype)
#print(itk.GetArrayFromImage(fixed_image_2))
#itk.imwrite(fixed_image_2, "fixed_image_2.png")
#np_array = image_list_slice[0].pixel_array
#print('Compute Metrik:')
#ComputeMetrik(image_list_slice[0].pixel_array,image_list_slice[3].pixel_array)
#itk_np = itk.GetImageFromArray(np.ascontiguousarray(np_array))

#region = itk_np.GetLargestPossibleRegion()
#size = region.GetSize()
#print(f"ITK image data size after convesion from NumPy = {size}\n")
#output_file_name = "fixed_image_3.png"
#itk.imwrite(itk_np, output_file_name)

#output_filename = "testPython2.png"

"""pixel_type = itk.UC
dimension = 2
image_type = itk.Image[pixel_type, dimension]

start = itk.Index[dimension]()
start.Fill(0)

size = itk.Size[dimension]()
size[0] = 200
size[1] = 300

region = itk.ImageRegion[dimension]()
region.SetIndex(start)
region.SetSize(size)

image = image_type.New(Regions=region)
image.Allocate()"""

#itk.imwrite(image, output_filename)
