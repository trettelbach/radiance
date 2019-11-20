import astropy.io.fits as pyfits
import torch.utils.data as data
import numpy as np
import pandas as pd
import mytransforms
import torchvision.transforms as transforms
import torch

# data transformation and gmentation needed for training the model
augment = transforms.Compose([
#    mytransforms.CenterCropFits(300),
    mytransforms.RandomHorizontalFlipFits(),
    mytransforms.RandomVerticalFlipFits(),
    mytransforms.ToTensorFits()
])

# data transformation for validating and testing the model
transform = transforms.Compose([
    mytransforms.CenterCropFits(4000),
#    mytransforms.Downsample(2),
    mytransforms.ToTensorFits()
])


# I need this to read my dataset. should be in another module?
class MyCustomDatasetFits(data.Dataset):
    # __init__ function is where the initial logic happens like reading a csv,
    # assigning transforms etc.
    def __init__(self, csv_path, transformation):
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # if the labels are known, the next column contains the label
#        if self.data_info.iloc[:, 1]
        print((self.data_info.shape[1]))
        if self.data_info.shape[1] == 2:
            self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.img_as_tensor = torch.Tensor()
        self.transformation = transformation

    # __getitem__ function returns the data and labels. This function is
    # called from dataloader like this
    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]

        data1 = pyfits.open(single_image_name, axes=2)
        data2 = data1[0].data.astype('float32')
#        print(len(data2.shape))
        if len(data2.shape) == 2:
            sh = data2.shape[0]
        elif len(data2.shape) == 4:
            sh = data2.shape[2]
        data3 = data2.reshape(4000, 4000, 1)

        img_mean = np.ndarray.mean(data3)
        img_std = np.std(data3)
        img_stand = (data3 - img_mean)/img_std

        # decide which transfomation should be used (train or test specific)
        if self.transformation == 'train':
            self.img_as_tensor = augment(img_stand)
        elif self.transformation == 'test':
            self.img_as_tensor = transform(img_stand)

        # Get label(class) of the image based on the cropped pandas column
        if self.data_info.shape[1] == 2:
            single_image_label = self.label_arr[index]
        else:
            single_image_label = 9999

        return self.img_as_tensor, single_image_label, single_image_name

    def __len__(self):
        return self.data_len
