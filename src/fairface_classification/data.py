import os
import json
import torch
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from skimage import io

warnings.filterwarnings('ignore')

def df_shuffling(df):
    """"
    Shuffles the rows of a dataframe.
    """

    indices = df.index.tolist()
    np.random.shuffle(indices)

    shuffled_df = df.iloc[indices].reset_index(drop=True)
    
    return shuffled_df

def train_test_split(dataframe, random_state=42, test_size=0.2):
    """
    Realization of train-test splitting function from scikit-learn library.
    """

    np.random.seed(random_state)
    shuffled_data = df_shuffling(dataframe)

    train_size = int(len(dataframe) * (1 - test_size))
    train_data = shuffled_data[:train_size]
    test_data = shuffled_data[train_size:]

    return train_data, test_data

def write_json(obj:dict, file:str):

    with open(file, 'w') as outfile:
        json.dump(obj, outfile)

    return None

def read_json(file):

    with open(file, 'r') as inp:
        json_object = json.load(inp)

    return json_object

def check_train_mode_off(loader, loader_type):
    assert_msg = f'First turn on the training mode in your {loader_type} dataloader with this snippet YourDataLoaderName.dataset.train()'

    assert loader.dataset.train_mode == True, assert_msg

    return True

class FairFaceDataset(Dataset):
    """Fair Face dataset."""

    def __init__(self, csv_file, root_dir, encoders:dict, train:bool, transform=None) -> None:
        """Arguments:
                csv_file (string): Path to the csv file with annotations and filenames.
                root_dir (string): Root directory of the project.
                images_dir (string): Directory that contains train and val dirs.
                transform (callable, optional): Optional transform to be applied 
                    on a sample.
        """

        # initialized for convenient observance of labels of the dataset 
        self.train_mode = True

        self.root_dir = root_dir
        self.images_dir = 'data'
        self.images_frame = pd.read_csv(os.path.join(self.root_dir, self.images_dir, csv_file))

        self.age_encoder = encoders['age']
        self.gender_encoder = encoders['gender']
        self.race_encoder = encoders['race']

        self.transform = transform

        train_test_data = read_json('../train_test_split.json')
        train_data, test_data = train_test_data['train'], train_test_data['test']

        if train:
            self.images_frame = self.images_frame[self.images_frame['file'].isin(train_data)]
        else:
            self.images_frame = self.images_frame[self.images_frame['file'].isin(test_data)]

    def _get_labels(self, index):
        age = self.images_frame.iloc[index]['age']
        gender = self.images_frame.iloc[index]['gender']
        race = self.images_frame.iloc[index]['race']

        if self.train_mode:
            age = self.age_encoder[age]
            gender = self.gender_encoder[gender]
            race = self.race_encoder[race]

        return {'age' : age, 'gender' : gender, 'race' : race}

    def __len__(self):
        return len(self.images_frame)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        image_name = os.path.join(self.root_dir, self.images_dir, self.images_frame.iloc[index, 0])
        image = io.imread(image_name)

        labels = self._get_labels(index=index)

        sample = {'image' : image, **labels}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def train(self):
        self.train_mode = True

    def view(self):
        self.train_mode = False
    

class CustomNormalizer(torch.nn.Module):
    """Normalizes image"""

    def __init__(self, mean, std) -> None:
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, sample):
        image = sample['image']

        for channel in range(image.shape[0]):
            image[channel] = image[channel] - self.mean[channel]
            image[channel] = image[channel] / self.std[channel]

        sample = {'image' : image, **{k:v for (k, v) in sample.items() if k != 'image'}}

        return sample
    
class CustomRandomVerticalFlipper:

    """Randomly vertically flip the image"""

    def __init__(self, prob) -> None:
        self.prob = prob

    def __call__(self, sample):
        image = sample['image']
        
        if np.random.random() <= self.prob:
            image = torch.flip(image, dims=[2])

        sample = {'image' : image, **{k:v for (k, v) in sample.items() if k != 'image'}}

        return sample
    
class ToTensor:

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))

        sample = {'image' : torch.from_numpy(image), **{k:v for (k, v) in sample.items() if k != 'image'}}

        return sample
    