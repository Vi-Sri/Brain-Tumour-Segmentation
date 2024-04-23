import cv2
import numpy as np
import nibabel as nib

label_path = 'Task01_BrainTumour/labelsTr/BRATS_001.nii.gz'
label = nib.load(label_path)
label = label.get_fdata()

unique_labels = np.unique(label[:,:,:])
print(unique_labels)

label_jpg_path = 'train/label/BRATS_471/BRATS_471_58.jpg'
label_jpg = cv2.imread(label_jpg_path, cv2.IMREAD_GRAYSCALE)
unique_labels_jpg = np.unique(label_jpg)
print(unique_labels_jpg)

