import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm

class BRATSDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        samples = []
        image_files = sorted(os.listdir(self.image_dir))
        print("Loading samples...")
        for image_file in tqdm(image_files):
            image_path = os.path.join(self.image_dir, image_file)
            label_path = os.path.join(self.label_dir, image_file)

            image = nib.load(image_path).get_fdata()
            label = nib.load(label_path).get_fdata()

            for i in range(label.shape[2]):
                if np.any(label[:, :, i] > 0):  
                    samples.append((image_path, label_path, i))
        print("Loaded ", len(samples), "samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_path, slice_index = self.samples[idx]
        image = nib.load(image_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # Extract specific slices
        image_slice = image[:, :, slice_index]
        label_slice = label[:, :, slice_index]

        image_slice = torch.Tensor(image_slice).unsqueeze(0)  # Adding channel dimension
        label_slice = torch.Tensor(label_slice).unsqueeze(0)

        if self.transform:
            image_slice = self.transform(image_slice)
            label_slice = self.transform(label_slice)

        return image_slice, label_slice
    
def main():
    image_dir = 'Task01_BrainTumour/imagesTr'  
    label_dir = 'Task01_BrainTumour/labelsTr'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-20,20),translate=(0.1,0.1),
                                 scale=(0.9,1.1), shear=(-0.2,0.2)),
        transforms.ElasticTransform(alpha=720., sigma=24.)])

    dataset = BRATSDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels.shape)
        plt.imshow(images[0, 0, :, :], cmap='gray')  # Show an image slice
        plt.imshow(labels[0, 0, :, :], cmap='jet', alpha=0.5)  # Overlay label slice
        plt.savefig('example.png')
        break  # Remove or comment this to process the entire dataset

if __name__ == '__main__':
    main()

