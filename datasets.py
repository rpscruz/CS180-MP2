import os
import nibabel
import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation as seg
from skimage.exposure import histogram
import cv2

dir_path = os.getcwd()

scans = np.ndarray([512, 512, 0])
'''
print("scan1")
scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(1).rjust(2, '0'))
scan = nibabel.load(scan_filename).get_data()
print(scans)
print(scan)

print(scans.shape)
print(scan.shape)
scans = np.stack([scans, scan])
print(scans.shape)
print(scan.shape)


print("scan2")
scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(2).rjust(2, '0'))
scan = nibabel.load(scan_filename).get_data()
print(scans)
print(scan)

print(scans.shape)
print(scan.shape)
scans = np.stack([scans, scan])
print(scans.shape)
print(scan.shape)


'''

# masks = np.ndarray([])

# test = np.ndarray([1, 3, 3])
print(scans)

#np.append(scans, test, axis = 0)

# print(scans)

# print(scans)

def get_max_depth():
    max_depth = 0
    for i in range(0 + 1, 40 + 1): # 40 + 1): # 1-indexed data
        mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        mask = nibabel.load(mask_filename)

        mask_data = mask.get_data().T

        if mask_data.shape[0] > max_depth:
            max_depth = mask_data.shape[0]
    return max_depth


from scipy.ndimage import zoom
def load_scans(samples_len, max_depth):
    scans = np.zeros((512, 512, 284))
    for i in range(0 + 1, samples_len + 1): # 40 + 1): # 1-indexed data
        
        print(str(i).rjust(2, '0'))
        
        scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        # mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))

        print("load_scans: loaded file")
        scan = nibabel.load(scan_filename)
        # mask = nibabel.load(mask_filename)

        print("load_scans: loaded data")
        scan_data = scan.get_data().T
        # mask_data = np.where(mask.get_data().T != 0, 1, 0)

        print("load_scans: computing scale to uniformity")
        scale = max_depth/scan_data.shape[0]
        # scale = max_depth/mask_data.shape[0]

        # scaled
        print("load_scans: rescaling to uniformity")
        print("scan was", scan_data.shape)
        scan_data = zoom(scan_data, (scale, 1, 1))
        # mask_data = zoom(mask_data, (scale, 1, 1))

        print("load_scans: final output scale is")
        print(scan_data.shape)
        # scans = np.stack([scans, scan])
        print("single was", scan.shape, "collated is", scans.shape)
        '''
    print(scans.shape)
    print(masks.shape)

    return np.array(scans), np.array(masks)
    '''




max_depth = get_max_depth()
print("the max is", max_depth)
load_scans(3, max_depth)


'''

# print(mask_data_transposed)
# plt.imshow(mask_data_transposed[200])
# plt.show()

# TRY THIS
# https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

# https://datascience.stackexchange.com/questions/28636/how-to-create-3d-images-from-nii-file
'''