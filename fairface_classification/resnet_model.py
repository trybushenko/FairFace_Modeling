import torch.nn as nn

from fairface_classification import ann_layers

class FairFaceResNet(nn.Module):
    def __init__(self, resnet) -> None:
        super(FairFaceResNet, self).__init__()

        self.resnet = resnet
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        for param in self.resnet.parameters():
            param.requires_grad = True
        
        self.fc1 = nn.Sequential(nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 9))
        self.fc2 = nn.Sequential(nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 2))
        self.fc3 = nn.Sequential(nn.Flatten(), 
                                 ann_layers.ANN_Blocks(in_features=2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(16, 7))

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(out.size(0), -1)

        age = self.fc1(out)
        gender = self.fc2(out)
        race = self.fc3(out)

        return {'age_pred' : age, 'gender_pred' : gender, 'race_pred' : race}
    


