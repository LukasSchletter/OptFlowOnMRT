# General tasks

 - Optimize the registration process. It should be done in 3d 
 - We might want to include the phase images
 - We might want to use ideas from the paper: Image Morphing in Deep Feature Spaces: Theory and     Applications


# Tasks for Lukas

3d registration!

 - files for registration
    + clean up code
    + do some experiments
      * Thin plate splines 
      * Smooth the images --> Learn transformation --> apply transformation on original images

- check out parameter map
- switch to elastix? Do you mean transformix?


 
 - making the comparison of all images
 - experiments with different registration metrices
 - print and read Image Morphing in Deep Feature Spaces: Theory and Applications

 Registration towards the next image?
 Differenz towards the next image and then plus?

 Make Visualization.

# Focus on Magnitude Images: The current pipeline primarily processes magnitude images, with phase images not included.

# Registration Framework: SimpleITK and Elastix are used for multi-stage registration, but the results might require refinement for precise alignment over time, especially for cardiac motion.

# Optical Flow Integration: RAFT is identified as the intended optical flow algorithm, but its application is still in progress, pending improvements in registration.


# Suggestions:

    - do you have Preprocessing and Normalization methods?
    Enhance registration accuracy (multi-stage with deformable methods and temporal smoothing).
    Validate registration thoroughly before optical flow computation.
    
    Explore if VoxelMorph can be used
    Is Total Deep Variation part of your approach?

# Approach
 - Make jupyter notebook for visualization and metric values
    + Organize and document scripts for clarity and reproducibility. I recommend but no needed, a jupyter notebook with visualization stage-by-stage.
 - Make a file computing the metric values
 - Do Registration for:
    + bspline (5, 2000)
    + groupwise
    + 3d (3d on registred slices)
    + Normalize the images
    + smoothing?
  - Write values into the notebook: Also pictures for picture
  - Do differences "like flow"

# Preprocessing
 - Chambolle`s Algorithm for noising


#  more todos
   - visualization of the flow --> Read me of voxelmorph
   - groupwise registration (3d)
   - 4d registration
   - normalize registration/smoothing
   - write mail
   - smooth images and apply to original data
   - Metric value abspeichern

# Visualization.py and 3d_Visualization
   - 2d Do 3 nebeneinander fixed reg, moved
   - mach raft und differenz für 3d und 2d
   - schicke mail