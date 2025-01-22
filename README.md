# OptFlowOnMRT
Compute the optical flow on registrated MRT images

# Data

Data is saved in the folder 4dflow_KI

Data contains images of 3 childeren

The most interesting series are the phaseimages and the magnitude images.

phaseimages might join the party but for now we are working with magnitude images of one patient just to
see if the hole concept works.

Therefore we are looking for one serie that contains the magnitude images of one patient.

readingDICOM.py will create a folder 'results_slices', where all the 110 slices à 25 images of this serie are saved in npy format for convenience. Note that some information are therefor lost, i.e. age of the child. The code is a bit messy. I cleaned it up somewhat but don't get confused if some unused lists are still there or stuff like that.

# Registration

The current main file for this is sITK_Reg.py. It might be good to formulate files for different Registration methods. 

Since the data folder is dependend on the current date and time you need to change the directory_path (line 51).

Registration manual: elastix-5.2.0-manual.pdf which can be found online
For quick introduction on SITK: https://simpleelastix.readthedocs.io/GettingStarted.html

Transformix might be execuded on smoothed images and apllied to original images.



# Differenz 

The main idea includes a registration process so the main structures of the images almost allign. The small structures will be to small to be registrated. Therefore the flow is measurable with RAFT or a Differenz image. 

differences.py produces these images. 

These images are nice for a quick checkup whether the registration is good enough (which is not the case right now)

# RAFT

RAFT is execuded on calci. I might include the code later if this is necessary. 
The main file however is R.py which is in the repo.

Paper: https://arxiv.org/pdf/2003.12039

# Other

+ movie.py

You can create a movie for demonstration purposes. Just adjust pathlist (line 37) to the folder with the images you want to put in the movie, the name of the movie (which would also be the path) ( line 39) and 
make sure you change if necessary the format of the images (line 19).

# Reading about MRT:
https://www.radiologie-luisenplatz.de/was-ist-mrt
https://www.youtube.com/watch?v=9PztQ3xoVOk

# Visualization
    Visualization.py gets you the Visualization for 2d images

    For 3d images use 3d_Visualization.py. Note that you first have to slice the image in 2d. Use split_3d.py for this purpose and make sure the paths are matching.

    3d fixed,reg,mov in elastix_3d/Visu_7_to_12
    2d fixed,reg,mov in elastix_bspline_gridspacing/Visualization

    Differenz 2d in Differenz/elastix_bspline_gridspacing
    Differenz 3d in

    Zeige mir die Maske der 3d images