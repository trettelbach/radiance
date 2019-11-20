# RADIAnCE - RADio-Image Artifact Classification nEtwork

This application is an image classification tool that aims at classifying flux density images observed by the MWA telescope. The classes specify the suitability of the images' incorporation into a sky survey. The classifier is based on a convolutional neural network and uses the PyTorch framework. 

The network's initial aim was to identify images from the GLEAM and GLEAM-X dataset that showed artifacts from either RFI or from undeconvolved sources in the sidelobes of the telescope's primary beam (cf. Rettelbach & Hurley-Walker 2020, in prep). The model's architecture is therefore optimized for solving this specific problem. To use this architecture for solving other problems via transfer learning, certain specifications and hyperparameters might need to be adapted. 

For each fits-file, the classifier will output probabilities for each target class. These results will then be added to the respective image's fits header file under a new keyword. If desired, the code will also output statistics for the classification, such as losses and the certainty of a prediction.


## Installation:
In order to run and make use of all features, the following libraries need to be installed:
- ```pytorch```
- ```astropy```
- ```glob2```
- ```pandas```
- ```PIL```
- ```scikit-learn```
- ```scikit-image```

However, there is a Docker image which also includes all necessary libraries.
A container from this image can be built with:

```docker pull trettelbach/pytorch_astro```

This eliminates the need to install the necessary libraries individually and keeps the python 
environment clean.

For guides on installing Docker, please refer to the [official Docker 
documentation](https://docs.docker.com/)



## Usage:
The repository contains three scripts which make use of the neural network. All scripts are 
coded with python and are conceived as command-line applications.
### Testing/Validating/Using only

The script ```scripts/cnn_gleam_inference.py``` loads the saved weights of the pre-trained network and runs the model on a validation/testing/unknown dataset. The fits header file will be updated to contain information on the classifications made by the network. If desired (if STATS set to 1, see below), it will also output text files with information on:

- true classes for testing and training (per image per epoch)
- predicted classes for testing and training (per image per epoch)
- testing and training accuracy (average per epoch)
- testing and training certainty (per image per epoch)
- testing and training losses (per image per epoch)
- time needed for each prediction/prediction and backpropagation testing and training (per image per epoch)


It can be run with the following:

```docker run trettelbach/pytorch_astro:latest python <PATH/TO/INFERENCE.py> <PATH/TO/THE/DIRECTORY/FOR/OUTPUTS/> <'PATH/TO/DIRECTORY/OF/CSV-FILE'>  <STATS>```

```<STATS>``` determines if the further statistics/outputs of the run shall be output as well. It can either assume the values 0 (no statistics) or 1 (statistics). **When there is no information on the true label, statistics cannot be output.**

When running the application on HPC systems, such as clusters by the Pawsey Supercomputing Centre, [*Shifter*](https://github.com/NERSC/shifter) will need to be used instead of Docker. The job script *slurm-scripts/cnn.slurm* represents the further necessary specifications to run the network on the clusters from the Pawsey Supercomputing Centre. 

Please note that in order to run the pre-trained network, it is necessary to have the information on the pre-trained weights. These are stored in a file ca. 270 MB in size, making them too large for GitHub. If you would like access to this file, please contact me via tabea dot rettelbach at awi dot de.

### Training only
The script ```scripts/cnn_gleam_train.py``` trains a neural model with the *CnnGleam* 
architecture and saves the model weights. If desired, it will also output statistics. If 
changes to hyperparameters (such as the number of epochs for training) need to be made, this 
must be adapted within the script.

It can be run with the following:

```docker run trettelbach/pytorch_astro:latest python <PATH TO cnn_gleam_train.py> <PATH TO THE 
DIRECTORY FOR OUTPUTS> <STATS>```

# Credits
This work was supported by resources provided by the Pawsey Supercomputing Centre with funding 
from the Australian Government and the Government of Western Australia.

# License
The code is published under the Academic Free License (AFL) v. 3.0.

