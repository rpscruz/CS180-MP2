import os
import nibabel
import numpy as np
import imageio

dir_path = os.getcwd()

def convert_3d_to_2d(samples_len):
    # scans = np.ndarray(shape=(0, 512, 512))
    # masks = np.ndarray(shape=(0, 512, 512))
    print(scans.shape)
    print(scans)


    for i in range(0 + 1, samples_len + 1): # 40 + 1): # 1-indexed data
        print(str(i).rjust(2, '0'))
        scan_filename = "{}/dataset_preprocess/train/scans/Patient_{}.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        mask_filename = "{}/dataset_preprocess/train/masks/Patient_{}_GT.nii.gz".format(dir_path, str(i).rjust(2, '0'))
        print(mask_filename)
        print("load_scans: loaded file")
        scan = nibabel.load(scan_filename)
        mask = nibabel.load(mask_filename)

        print("load_scans: loaded data")
        scan_data = scan.get_data()
        mask_data = mask.get_data()

        scan_data = scan_data.T
        mask_data = mask_data.T
        result = np.where(mask_data>0, 255, mask_data) 

        
        for slide in range(len(scan_data)):
            imageio.imwrite("data/scans/{}_{}.png".format(i, slide), scan_data[slide])
            imageio.imwrite("data/masks/{}_{}.png".format(i, slide), result[slide])
        

        print(scans.shape)
        print(scan_data.shape)
    return

   
# convert_3d_to_2d(40)

import glob

def load_scans():
    scans, masks = [], []
    for scan_path in glob.glob("{}/data/scans/*_*.png".format(dir_path)):
        scan = imageio.imread(scan_path)
        scans.append(scan)

    k = 0
    for mask_path in glob.glob("{}/data/masks/*_*.png".format(dir_path)):
        mask = imageio.imread(mask_path)
        masks.append(mask)

    scans = np.asarray(scans)
    masks = np.asarray(masks)
    return scans, masks