from torch import nn

class ANN_cycle(nn.Module):
    """
    ANN cycle that consists of linear layer, RelU and dropout ones.
    
    Args:
        in_features: number of input neurons in the current ANN cycle
        out_features: number of output neurons in the current ANN cycle
    """
    def __init__(self, in_features, out_features, dropout_val=0.3) -> None:
        super(ANN_cycle, self).__init__()
        self.block = nn.Sequential(nn.Linear(in_features, out_features),
                                   nn.ReLU(),
                                   nn.Dropout(dropout_val))
        
    def forward(self, x):
        return self.block(x)

class ANN_Blocks(nn.Module):
    """
    A stack of ANN cycles.

    Each ANN cycle consists of a linear layer, a ReLU activation function, and a dropout layer.

    Args:
        None
    """
    def __init__(self) -> None:
        super(ANN_Blocks, self).__init__()
        self.block = nn.Sequential(ANN_cycle(960, 512, 0.3),
                                 ANN_cycle(512, 256, 0.3),
                                 ANN_cycle(256, 128, 0.3),
                                 ANN_cycle(128, 64, 0.3),
                                 ANN_cycle(64, 32, 0.3),
                                 ANN_cycle(32, 16, 0.3))
        
    def forward(self, x):
        return self.block(x)
    
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

        self.fc1 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Flatten(), 
                                 ANN_Blocks(),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 9))
        self.fc2 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Flatten(), 
                                 ANN_Blocks(),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 2))
        self.fc3 = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                 nn.Flatten(), 
                                 ANN_Blocks(),
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