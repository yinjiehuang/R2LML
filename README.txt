=============================================
 Reduced-Rank Local Distance Metric Learning
 Yinjie Huang
 University of Central Florida
 2013
=============================================



=============================================
=========
CONTENTS:
=========

1. General Information
2. Requirements
3. Installation
4. Usage

=============================================

1. GENERAL INFORMATION

This software was written as the demo of RET project Summer 2013. It represents an implementation of
an automatic Eyes & Mouth detection algorithm and face morphing algorihm, including a complete graphical user interface (GUI). All rights belong to the author.



2. REQUIREMENTS

To run this software, you need to have the following components installed:
- Mathworks MATLAB
- Mathworks Image Processing Toolbox



3. INSTALLATION

This software doesn't require any installation. Just drop the files into a folder.



4. USAGE

To run the software, run the file 'Main.m' or type in 'Main' in the MATLAB command window. The script will take care of all the rest and start a graphical user interface. 

The basic usage is as follows:
- Open Two faces: Face 1 & Face 2
- Automatic detect Eyes and Mouth in each face
- The detection algorithm may fail. You could fix the positions of eyes or mouths. For example, after clicking button 'Left Eye', click the true eye area in the image and a box should pop out to tell you the true coordinates of your click. Finally, click "Left Eye" again to finish this fix action. Click "Done", you will see the detection results are changed.
- Morph the two faces together. Alpha value (between 0 and 1) is the crossing dissolve parameters.
- Show video of face morphing. And the video will be saved as ".avi" in the main folder. 
