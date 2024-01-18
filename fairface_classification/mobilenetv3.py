import torch

from torch import nn

from fairface_classification import ann_layers
 
# FairFaceMobileNet is a neural network model for age, gender, and race classification.
# It is based on the MobileNet architecture, with additional ANN blocks added to the end.
#
# The model takes an image as input and outputs three predictions:
# - age_pred: a 9-dimensional vector representing the probability of each age class
# - gender_pred: a 2-dimensional vector representing the probability of male and female
# - race_pred: a 7-dimensional vector representing the probability of each race

class FairFaceMobileNet(nn.Module):
    def __init__(self, MobileNet) -> None:
        """
        Initializes a new FairFaceMobileNet model.

        Args:
            MobileNet: a pre-trained MobileNet model
        """
        super(FairFaceMobileNet, self).__init__()
        self.mobilenet = MobileNet
        for param in self.mobilenet.parameters():
            param.requires_grad = True

        out_ftrs_num = self.mobilenet.features[-1][0].out_channels
        self.fc1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=out_ftrs_num),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 9))
        self.fc2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=out_ftrs_num),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 2))
        self.fc3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=out_ftrs_num),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 7))

    def forward(self, x):
        """
        Args:
            x: an image tensor of shape (batch_size, channels, height, width)

        Returns:
            a dictionary containing the three predictions:
                - age_pred: a 9-dimensional vector representing the probability of each age class
                - gender_pred: a 2-dimensional vector representing the probability of male and female
                - race_pred: a 7-dimensional vector representing the probability of each race
        """

        out = self.mobilenet.features(x)

        age = self.fc1(out)
        gender = self.fc2(out)
        race = self.fc3(out)

        return {'age_pred' : age, 'gender_pred' : gender, 'race_pred' : race}