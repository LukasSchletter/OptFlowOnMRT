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

