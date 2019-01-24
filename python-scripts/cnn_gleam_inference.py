#import PIL
import torch
import torch.nn as nn
import astropy.io.fits as pyfits
import torch.utils.data as data
from torch.autograd import Variable
import cnn_gleam
import myreader
import time
import sys
import argparse


def main():
    """
    Loads the saved weights of the CnnGleam
    model to run the network algorithm on a
    unlabeled dataset.

    Fits headers will be updated to contain information
    on the classifications made by the network
    Also prints some statistics, if desired.
    """

    # read path to csv and if user wants output statistics from command line argument
    parser = argparse.ArgumentParser(description='Run the GleamNet for Classification')
    parser.add_argument('path', metavar='PATH', type=str, nargs='?', help='path to the csv with file locations')
    parser.add_argument('statistics', metavar='STATS', type=int, nargs='?', help='determine if statistics should be calculated')
    args = parser.parse_args()

    # set path to data, number of dataset labels, version number of network run
    PATH = args.path
    NLABEL = 4
    VER = 1.0

    start_time = time.time()

    # check if GPU available, and set device accordingly
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # does the user want statistics? (read from parser)
    if args.statistics == 0:
        stats = False
        sys.stdout.write('updating fits headers only\n')
    else:
        stats = True
        sys.stdout.write('updating fits headers and producing further statistics\n')

    # load testing set, data augmentation
    fits_test = myreader.MyCustomDatasetFits(PATH + 'testing.csv', transformation='test')
    testloader = torch.utils.data.DataLoader(fits_test, shuffle=True, num_workers=1)

    # initialize the architecture for the model
    model = cnn_gleam.CnnGleam(NLABEL).to(device)
    # load saved weights to the model
    model.load_state_dict(torch.load(PATH + 'gleamnet31'))

    # define criterion (loss function)
    criterion = nn.CrossEntropyLoss()

    # prepare some lists for statistical evaluation
    if stats:
        test_pred = []
        test_cert = []
        test_loc = []
        test_time_im = []

    # set model to evaluation mode
    model.eval()

    correct_test = 0

    # iterate through the testing set
    for images, labels, location in testloader:
        start_test_im = time.time()
        # send tensors to GPU/CPU
        if use_gpu:
            images = Variable(images.float().cuda(), requires_grad=True, volatile=False)
        else:
            images = Variable(images.float(), requires_grad=True, volatile=False)

        # run the model and apply SoftMax at the end
        outputs = model(images)
        m = nn.Softmax(dim=1)
        probs = m(outputs[0])
        # get predictions
        max_values, max_indices = torch.max(probs[0], 0)

        # determine prediction duration per image
        if stats:
            end_test_im = time.time()
            test_time_im.append(end_test_im - start_test_im)

        # write the results to the fits-header
        hdulist = pyfits.open(location[0])
        img_header = hdulist[0].header
        img_header.set('cnnver', VER)
        img_header.set('good', probs[0][0].item())
        img_header.set('rfi', probs[0][1].item())
        img_header.set('sis', probs[0][2].item())
        img_header.set('rfisis', probs[0][3].item())
        hdulist.writeto(location[0], overwrite=True)
        hdulist.close()

        # calculate some statistics per image
        if stats:
            test_loc.append(location)
            test_pred.append(max_indices.item())
            test_cert.append(max_values.item())

    # write outputs to .txt files for later evaluation
    if stats:
        with open(PATH + 'test_pred.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_pred)
        with open(PATH + 'test_cert.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_cert)
        with open(PATH + 'test_loc.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_loc)
        with open(PATH + 'test_time.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in test_time_im)

    sys.stdout.write('Duration of testing (in s): %.2f\n' % (time.time() - start_time))


if __name__ == "__main__":
    main()

