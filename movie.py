import os
import moviepy.video.io.ImageSequenceClip
import moviepy.video.VideoClip
from PIL import Image
from moviepy.editor import *
from PIL import Image
from PIL import ImageDraw



def create_movie(name, pathlist):
    
    
    fps=1
    
    image_files = [os.path.join(image_folder,img) #gets path to the specific image
                for image_folder in pathlist
                for img in os.listdir(image_folder) #Return a list containing the names of the entries in the directory given by path.
                if img.endswith(".png")]
    print(len(image_files))
    print(image_files)
    
    image_li = []
    for filename in image_files: 
        number = int(os.path.splitext(os.path.basename(filename))[0])
        print(number)
        entity = [number, filename]
        image_li.append(entity)
    print('length of list = ' + str(len(image_li)))
    sorted_list = sorted(image_li)
    im_sort = [i[1] for i in sorted_list]

                  
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(im_sort, fps=fps)
    clip.write_videofile(name)

pathlist = ['movie_raft/vergleich'] 

create_movie('vergleich.mp4', pathlist )
