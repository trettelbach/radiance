import torch.nn as nn

class CnnGleam(nn.Module):
    def __init__(self, nlabel):
        super(CnnGleam, self).__init__()
        self.nlabel = nlabel

        # all building blocks for the network (3 hidden layers with Conv > ReLU > MaxPool each)
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=15, padding=7)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.MaxPool2d(3)

        self.layer4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=9, padding=4)
        self.layer5 = nn.ReLU()
        self.layer6 = nn.MaxPool2d(3)

        self.layer7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.layer8 = nn.ReLU()
        self.layer9 = nn.MaxPool2d(3)

        # 3 fully connected layers with dropout initialization
        self.fc1 = nn.Linear(700928, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, self.nlabel)
        self.dropout = nn.Dropout(p=0.1)

        # allow parallel processing
        self.fc1 = nn.DataParallel(self.fc1)
        self.fc2 = nn.DataParallel(self.fc2)
        self.fc3 = nn.DataParallel(self.fc3)

    def forward(self, x):
        ''' defines the architecture and 
        exact layer sequence of the model
        '''
        out = self.layer1(x)

        # for visualizing the filters of the first layer
        a = self.layer1.weight
        self.b = a.cpu().detach().numpy()
        self.weights1.append(self.b)
        # for visualizing the feature maps after the Conv operation of the first layer
        self.out4pic1 = out.cpu().detach().numpy()

        out = self.layer2(out)
        out = self.layer3(out)

        out = self.layer4(out)
        # for visualizing the filters of the second layer
        c = self.layer4.weight[0][0]
        d = c.cpu().detach().numpy()

        out = self.layer5(out)
        out = self.layer6(out)

        out = self.layer7(out)
        # for visualizing the filters of the third layer
        e = self.layer7.weight
        self.f = e.cpu().detach().numpy()
        self.weights9.append(self.f)

        out = self.layer8(out)
        out = self.layer9(out)

        # for visualizing the feature maps after the Pool operation of the third layer
        self.out4pic9 = out.cpu().detach().numpy()

        # flatten layers
        out = out.view(out.size(0), -1)

        # fully connected layers at the end (with dropout)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out, self.b, self.out4pic1, self.f, self.out4pic9
