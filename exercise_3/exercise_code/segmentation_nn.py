"""SegmentationNN"""
import torch
import torch.nn as nn


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        #the output layer should be 1x1x23
        #in_channels is the color, eg. if rgb 3, if black 1.
        from torchvision import models
        self.features = models.vgg16(pretrained=True).features
        self.model = nn.Sequential(nn.Dropout(),
                                   nn.Conv2d(512, 4096, kernel_size=1, padding=0),
                                   nn.ReLU(),
                                   nn.Dropout(),
                                   nn.Conv2d(4096, 2048, kernel_size=2, padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(2048, num_classes, kernel_size=2, padding=0),
                                   nn.Upsample(scale_factor=20),
                                   nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
                                   nn.Upsample(scale_factor=2.4)
                                  )
        """self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=10, stride=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )"""
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x = self.features(x)
        x = self.model(x)
        #x = x.view(x.size()[0], -1)
        #x = self.classifier(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
