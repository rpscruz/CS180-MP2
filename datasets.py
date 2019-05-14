import os
import nibabel
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as seg
from skimage.exposure import histogram
from statistics import mean

dir_path = os.getcwd()

'''
def get_ave_depth():
    ave_depth = []
    for i in range(0 + 1, 40 + 1): # 40 + 1): # 1-indexed data
        print("get_ave_depth:", i)
        mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        mask = nibabel.load(mask_filename)

        ave_depth.append(mask.get_data().T.shape[0])
    return round(mean(ave_depth))
'''

from scipy.ndimage import zoom
def load_scans(samples_len, ave_depth):
    scans = []
    masks = []
    for i in range(0 + 1, samples_len + 1): # 40 + 1): # 1-indexed data
        print(str(i).rjust(2, '0'))
        scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))

        print("load_scans: loaded file")
        scan = nibabel.load(scan_filename)
        mask = nibabel.load(mask_filename)

        print("load_scans: loaded data")
        scan_data = scan.get_data()
        mask_data = mask.get_data()

        print("load_scans: computing scale to uniformity")
        scale = ave_depth/scan_data.shape[2]

        # scaled
        print("load_scans: rescaling to uniformity")
        scan_data = zoom(scan_data, (1, 1, scale))
        mask_data = zoom(mask_data, (1, 1, scale))

        mask_data = np.where(mask_data != 0, 1, 0)

        # appending data to data lists
        scans.append(scan_data)
        masks.append(mask_data)

        print("load_scans: final output scale is")
        print(scan_data.shape)
        print(mask_data.shape)

    
    # print(scans.shape)
    #print(masks.shape)
    #return 
    return np.stack(scans), np.stack(masks)

'''
# ave_depth = get_ave_depth()
# memoize: 186.
ave_depth = 186

scans, masks = load_scans(3, 186)
print("final scans are", scans.shape)
print("final masks are", masks.shape)

'''

