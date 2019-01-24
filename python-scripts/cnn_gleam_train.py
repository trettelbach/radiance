import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from cnn_gleam import CnnGleam
import astropy.io.fits as pyfits
import myreader
import time
import sys
import argparse
from PIL import Image


def main():
    """ Trains a neural model with the CnnGleam architecture,
    saves the model weights,
    and prints some statistics, if desired
    """

    # read path to csv and if user wants output statistics from command line argument 
    parser = argparse.ArgumentParser(description='Train and test the GleamNet for Classification')
    parser.add_argument('path', metavar='PATH', type=str, nargs='?', help='path to the csv with file locations')
    parser.add_argument('statistics', metavar='STATS', type=int, nargs='?', help='determine if statistics should be calculated')
    args = parser.parse_args()
    print(args.statistics)

    # set path to data, number of dataset labels,
    # number of epochs to train, initial learning rate
    PATH = args.path
    NLABEL = 4
    EPOCHS = 50
    INIT_LR = 5e-7
    VER = 1.0
    runn = 35

    stats = True

    # set time
    start_time = time.time()
    train_time_im = []

    # check if GPU available, and set device accordingly
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    # does the user want statistics? (read from parser)
    if args.statistics == 0:
        stats = False
        sys.stdout.write('training the model only\n')
    else:
        stats = True
        sys.stdout.write('training the model and producing further statistics\n')

    # load training set, data augmentation
    fits_train = myreader.MyCustomDatasetFits(PATH + 'training.csv', transformation='train')
    trainloader = torch.utils.data.DataLoader(fits_train, shuffle=True, num_workers=1)

    # make instance of the Conv Net, send to GPU if available
    model = CnnGleam(NLABEL)
    if use_gpu:
        model = model.cuda()

    # define criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

    # prepare some lists for statistical evaluation
    if stats:
        train_losses = []
        train_true = []
        train_pred = []
        train_cert = []
        train_loc = []
        train_acc = []

    # run the data through the network for x epochs
    for epoch in range(EPOCHS):
        # set model to training mode
        model.train()
        correct_train = 0

        # iterate through the trainig set
        for i, (images, labels, location) in enumerate(trainloader):
            start_train_im = time.time()
            # send tensors to GPU/CPU
            if use_gpu:
                labels = Variable(labels.cuda(), volatile=False)
                images = Variable(images.float().cuda(), requires_grad=False, volatile=False)
            else:
                labels = Variable(labels, volatile=False)
                images = Variable(images.float(), requires_grad=False, volatile=False)

            # clear gradients of all optimized tensors
            optimizer.zero_grad()

            # run the model and apply softmax at the end to get classification probabilities
            outputs = model(images)
            m = nn.Softmax(dim=1)
            probs = m(outputs[0])
            # get predictions
            max_values, max_indices = torch.max(probs[0], 0)

            # calculate loss
            train_loss = criterion(outputs[0], labels.squeeze(1).type(torch.cuda.LongTensor))

            # parameter update based on the current gradient
            train_loss.backward()
            optimizer.step()
  
            end_train_im = time.time()
            sys.stdout.write('correct: %d - predicted: %d at %.3f %% - image: %s\n' % (labels.item(), max_indices.item(), max_values.item() * 100, location))

            # calculate some statistics per image
            if stats:
                correct_train += (max_indices == labels).sum().item()
                train_losses.append(train_loss.item())
                train_true.append(labels.item())
                train_pred.append(max_indices.item())
                train_loc.append(location)
                train_cert.append(max_values.item())
                train_time_im.append(end_train_im - start_train_im)

        # get the overall accuracy
        if stats:
            train_acc.append(correct_train / float(len(trainloader)))
      
    # write outputs to .txt files for later evaluation
    if stats == True:
        with open(PATH + 'train_losses' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_losses)
        with open(PATH + 'train_true' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_true)
        with open(PATH + 'train_pred' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_pred)
        with open(PATH + 'train_cert' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_cert)
        with open(PATH + 'train_loc' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_loc)
        with open(PATH + 'train_acc' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_acc)
        with open(PATH + 'train_time' + str(runn) + '.txt', 'w') as ff:
            ff.writelines("%s\n" % elem for elem in train_time_im)

    # save the model weights
    torch.save(model.state_dict(), PATH + 'gleamnet' + str(runn))

    sys.stdout.write('Duration of training (in s): %.2f\n' % (time.time() - start_time))


if __name__ == "__main__":
    main()
