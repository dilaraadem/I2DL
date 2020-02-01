import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        #in_channels is the color, eg. if rgb 3, if black 1.
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2,2),
                                  nn.Dropout(0.2),
                                  nn.Conv2d(32, 64, 3, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2,2),
                                  nn.Dropout(0.3),
                                  nn.Conv2d(64, 128, 3, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(2,1),
                                  nn.Dropout(0.4),
                                  nn.Conv2d(128, 256, 2, 1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(1,1),
                                  nn.Dropout(0.5)
                                 )
        self.fc = nn.Sequential(nn.Linear(9216, 256, bias=True),
                                nn.ReLU(),
                                nn.Dropout(),
                                #nn.Linear(3000, 1500, bias=True),
                                #nn.ReLU(),
                                #nn.Dropout(),
                                nn.Linear(256, 30, bias=True),
                                nn.Tanh()
                               )
        # without tanh it goes weird.

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
