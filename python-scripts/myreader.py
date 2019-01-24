import astropy.io.fits as pyfits
import torch.utils.data as data
import numpy as np
import pandas as pd
import mytransforms
import torchvision.transforms as transforms
import torch

# data transformation and augmentation needed for training the model
augment = transforms.Compose([
    mytransforms.RandomHorizontalFlipFits(),
    mytransforms.RandomVerticalFlipFits(),
    mytransforms.ToTensorFits()
])

# data transformation for validating, testing, and inference the model
transform = transforms.Compose([
    mytransforms.CenterCropFits(4000),
#    mytransforms.Downsample(2),
    mytransforms.ToTensorFits()
])


# custom Dataset reader to read the fits images from a csv-file
# the first column has to contain the entire image paths
# the second column is for the assigned label (must be continuous integers starting from 0)
class MyCustomDatasetFits(data.Dataset):
    def __init__(self, csv_path, transformation):
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # the rest contain the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 1:])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.img_as_tensor = torch.Tensor()
        self.transformation = transformation

    # returns the data and labels. It is called from the dataloader
    def __getitem__(self, index):
        # Get image name from the dataframe
        single_image_name = self.image_arr[index]

        data1 = pyfits.open(single_image_name, axes=2)
        data2 = data1[0].data.astype('float32')
        if len(data2.shape) == 2:
            sh = data2.shape[0]
        elif len(data2.shape) == 4:
            sh = data2.shape[2]
        data3 = data2.reshape(sh, sh, 1)

        # normalize image data
        img_mean = np.ndarray.mean(data3)
        img_std = np.std(data3)
        img_stand = (data3 - img_mean)/img_std

        # apply transforms to dataset
        if self.transformation == 'train':
            self.img_as_tensor = augment(img_stand)
        elif self.transformation == 'test':
            self.img_as_tensor = transform(img_stand)

        # Get label(class) of the image based on the dataframe column
        single_image_label = self.label_arr[index]

        return self.img_as_tensor, single_image_label, single_image_name

    def __len__(self):
        return self.data_len
