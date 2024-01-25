import os
import json
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset

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
    """Writes a JSON object to a file.

    Args:
        obj: The JSON object to write.
        file: The path to the file to write to.

    Returns:
        None.
    """
    with open(file, 'w') as outfile:
        json.dump(obj, outfile)

    return None

def read_json(file):
    """Reads a JSON file and returns the JSON object.

    Args:
        file: The path to the JSON file.

    Returns:
        A JSON object.
    """
    with open(file, 'r') as inp:
        json_object = json.load(inp)

    return json_object

def check_train_mode_off(loader, loader_type):
    """Checks whether the loader is in train mode.

    Args:
        loader: The loader to check.
        loader_type: The type of loader.

    Returns:
        A boolean indicating whether the loader is in train mode.
    """
    assert_msg = f'First turn on the training mode in your {loader_type} dataloader with this snippet YourDataLoaderName.dataset.train()'

    assert loader.dataset.train_mode == True, assert_msg

    return True

class FairFaceDataset(Dataset):
    """Fair Face dataset."""

    def __init__(self, csv_file, root_dir, encoders:dict, mode:str, transform=None) -> None:
        """Arguments:
                csv_file (string): Path to the csv file with annotations and filenames.
                root_dir (string): Root directory of the project.
                images_dir (string): Directory that contains train and val dirs.
                encoders
                mode (string): can "train", "test", "val"
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

        train_test_data = read_json('../train_test_val.json')
        train_data, test_data, val_data = train_test_data['train'], train_test_data['test'], train_test_data['val']

        if mode == 'train':
            self.images_frame = self.images_frame[self.images_frame['file'].isin(train_data)]
        elif mode == 'test':
            self.images_frame = self.images_frame[self.images_frame['file'].isin(test_data)]
        else:
            self.images_frame = self.images_frame[self.images_frame['file'].isin(val_data)]

    def _get_labels(self, index):
        """Gets the labels for the given index.

        Args:
            index: The index of the image.

        Returns:
            A dictionary containing the labels for the image.
        """
        age = self.images_frame.iloc[index]['age']
        gender = self.images_frame.iloc[index]['gender']
        race = self.images_frame.iloc[index]['race']

        if self.train_mode:
            age = self.age_encoder[age]
            gender = self.gender_encoder[gender]
            race = self.race_encoder[race]

        return {'age' : age, 'gender' : gender, 'race' : race}

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            The length of the dataset.
        """
        return len(self.images_frame)
    
    def __getitem__(self, index):
        """Gets an item from the dataset.

        Args:
            index: The index of the item to get.

        Returns:
            A dictionary containing the image and labels.
        """
        if torch.is_tensor(index):
            index = index.tolist()
        
        image_name = os.path.join(self.root_dir, self.images_dir, self.images_frame.iloc[index, 0])

        image = Image.open(image_name)

        labels = self._get_labels(index=index)

        if self.transform:
            image = self.transform(image)

        sample = {'image' : image, **labels}

        return sample
    
    def train(self):
        """Sets the dataset to train mode.

        Returns:
            None.
        """
        self.train_mode = True

    def view(self):
        """Sets the dataset to view mode.

        Returns:
            None.
        """
        self.train_mode = False

def get_dummy(enc_dict, label, device):
    """
    Converts a label to a one-hot encoded vector.

    Args:
        enc_dict: A dictionary mapping labels to their corresponding indices.
        label: The label to be converted.
        device: The device to be used.

    Returns:
        A one-hot encoded vector representing the label.
    """
    label_classes = len(enc_dict.values())
    label_array = torch.zeros((label.shape[0], label_classes))

    for (i, label_class) in enumerate(label):
        label_array[i][label_class] = 1

    label_array = label_array.to(device)
    return label_array

class Unnormalize(nn.Module):
    def __init__(self, mean, std):
        super(Unnormalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return self.unnormalize(x)

    def unnormalize(self, x):
        print(x.shape)
        # Assuming x is a PyTorch tensor
        return x * self.std[None, :, None, None] + self.mean[None, :, None, None]