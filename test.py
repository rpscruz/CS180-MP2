import os

import nibabel as nib
from nilearn import image
from nilearn import plotting

print("Path at terminal when executing this file")
# filename = os.getcwd() + "/dataset/test/"

# filename = os.getcwd() + "/dataset/train/Patient_01/Patient_01.nii.gz"
filename = os.getcwd() + "/dataset/train/Patient_01/GT.nii.gz"

smooth_img = image.smooth_img(filename, fwhm=None)
print(smooth_img)
display = plotting.plot_img(smooth_img)
display.savefig('pretty_brain.png')
display.close()
print(filename)

'''
plotting.plot_img(filename)
image.load_img(filename)
'''