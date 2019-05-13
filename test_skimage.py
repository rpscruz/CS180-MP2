import os
import nibabel
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as seg
from skimage.exposure import histogram

dir_path = os.getcwd()
images = []
masks = []

for i in range(0 + 1, 1 + 1): # 40 + 1): # 1-indexed data
    
    print(str(i).rjust(2, '0'))
    
    scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(i).rjust(2, '0'))
    mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))


    scan = nibabel.load(scan_filename)
    mask = nibabel.load(mask_filename)

    scan_data = scan.get_data().T

    mask_data = np.where(mask.get_data().T != 0, 1, 0)

    images.append(scan_data)
    masks.append(mask_data)

images_np = np.array(images)
'''

# print(mask_data_transposed)
# plt.imshow(mask_data_transposed[200])
# plt.show()

# TRY THIS
# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

# https://datascience.stackexchange.com/questions/28636/how-to-create-3d-images-from-nii-file
'''