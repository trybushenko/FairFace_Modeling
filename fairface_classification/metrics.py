import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
from tqdm import tqdm

def create_cm_df(cf, classes):
    """
    Creates a confusion matrix DataFrame.

    Args:
        cf: A confusion matrix.
        classes: A list of the classes in the confusion matrix.

    Returns:
        A confusion matrix DataFrame.
    """
    return pd.DataFrame(cf / np.sum(cf, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])

def create_cfs(ages, genders, races, mappings):
    """
    Creates a confusion matrix plot for each class.

    Args:
        ages: A dictionary of age labels and predictions.
        genders: A dictionary of gender labels and predictions.
        races: A dictionary of race labels and predictions.
        mappings: A dictionary mapping labels to their corresponding indices.

    Returns:
        A figure containing the confusion matrix plots.
    """
    age_cf = confusion_matrix(ages['true'], ages['pred'])
    gender_cf = confusion_matrix(genders['true'], genders['pred'])
    race_cf = confusion_matrix(races['true'], races['pred'])

    age_df = create_cm_df(age_cf, mappings['age_map'])
    gender_df = create_cm_df(gender_cf, mappings['gender_map'])
    race_df = create_cm_df(race_cf, mappings['race_map'])

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (24, 5))
    sns.heatmap(age_df, annot=True, ax=axes[0])
    sns.heatmap(gender_df, annot=True, ax=axes[1])
    sns.heatmap(race_df, annot=True, ax=axes[2])
    plt.close()
    return fig

def build_confusion_matrix(model, loader, mappings, device):
    """Builds a confusion matrix for the given model, loader, mappings, and device.

    Args:
        model: The model to evaluate.
        loader: The loader to use to load the data.
        mappings: A dictionary of mappings from labels to indices.
        device: The device to use.

    Returns:
        A tuple of three dictionaries: ages, genders, and races. Each dictionary contains two lists: pred and true.
    """
    ages = {'pred' : [], 'true' : []}
    genders = {'pred' : [], 'true' : []}
    races = {'pred' : [], 'true' : []}
    # Perform the full cycle of the iterating over the whole test dataloder and append predictions with labels to the vectors
    with torch.no_grad():   
          
        for batch in tqdm(loader):

            image, race = batch['image'].to(device), batch['race'].to(device)
            gender, age = batch['gender'].to(device), batch['age'].to(device)

            output = model(image)

            # Age accuracy
            _, age_predicted = torch.max(torch.softmax(output['age_pred'].data, dim=1), 1)
            # Race accuracy
            _, race_predicted = torch.max(torch.softmax(output['race_pred'].data, dim=1), 1)
            # Gender accuracy
            _, gender_predicted = torch.max(torch.softmax(output['gender_pred'].data, dim=1), 1)

            ages['pred'].extend(age_predicted.cpu())
            genders['pred'].extend(gender_predicted.cpu())
            races['pred'].extend(race_predicted.cpu())

            ages['true'].extend(age.cpu())
            genders['true'].extend(gender.cpu())
            races['true'].extend(race.cpu())

    # Create confusion matrix
    age_cf = confusion_matrix(ages['true'], ages['pred'])
    gender_cf = confusion_matrix(genders['true'], genders['pred'])
    race_cf = confusion_matrix(races['true'], races['pred'])

    # Create confusion matrix dataframe 
    age_df = create_cm_df(age_cf, mappings['age_map'])
    gender_df = create_cm_df(gender_cf, mappings['gender_map'])
    race_df = create_cm_df(race_cf, mappings['race_map'])

    # Plot confusion matrices for every head of the provided Neural Network
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (24, 5))
    sns.heatmap(age_df, annot=True, ax=axes[0])
    sns.heatmap(gender_df, annot=True, ax=axes[1])
    sns.heatmap(race_df, annot=True, ax=axes[2])
    plt.show()


    return ages, genders, races

def calc_measures(model, loader, device):
    """Calculates the accuracy, precision, recall, and f1-score for the given model and loader.

    Args:
        model: The model to evaluate.
        loader: The loader to use to load the data.

    Returns:
        A Pandas DataFrame containing the accuracy, precision, recall, and f1-score for each attribute (age, gender, and race).
    """
    # Initialize the dictionaries to store the predictions and labels.
    ages = {'pred' : [], 'true' : [] }
    genders = {'pred' : [], 'true' : [] }
    races = {'pred' : [], 'true' : [] }

    # Create a dictionary to store the measures.
    measures = {}

    # Iterate over the loader.
    with torch.no_grad():   
          
        for batch in tqdm(loader):
            # Move the images and labels to the device.
            image, race = batch['image'].to(device), batch['race'].to(device)
            gender, age = batch['gender'].to(device), batch['age'].to(device)
            
            # Make a prediction.
            output = model(image)
            # Get the predicted and true labels.
            _, age_predicted = torch.max(torch.softmax(output['age_pred'].data, dim=1), 1)
            _, race_predicted = torch.max(torch.softmax(output['race_pred'].data, dim=1), 1)
            _, gender_predicted = torch.max(torch.softmax(output['gender_pred'].data, dim=1), 1)

            # Add the predictions and labels to the dictionaries.
            ages['pred'].extend(age_predicted.cpu())
            genders['pred'].extend(gender_predicted.cpu())
            races['pred'].extend(race_predicted.cpu())

            ages['true'].extend(age.cpu())
            genders['true'].extend(gender.cpu())
            races['true'].extend(race.cpu())

    # Calculate the accuracy, precision, recall, and f1-score for each attribute.
    measures['age'] = [accuracy_score(ages['true'], ages['pred']), precision_score(ages['true'], ages['pred'], average='macro'), 
                       recall_score(ages['true'], ages['pred'], average='macro'), f1_score(ages['true'], ages['pred'], average='macro')]
    measures['gender'] = [accuracy_score(genders['true'], genders['pred']), precision_score(genders['true'], genders['pred'], average='macro'), 
                          recall_score(genders['true'], genders['pred'], average='macro'), f1_score(genders['true'], genders['pred'], average='macro')]
    measures['race'] = [accuracy_score(races['true'], races['pred']), precision_score(races['true'], races['pred'], average='macro'), 
                        recall_score(races['true'], races['pred'], average='macro'), f1_score(races['true'], races['pred'], average='macro')]
    # Return a Pandas DataFrame containing the measures.
    return pd.DataFrame(measures, index=['accuracy', 'precision', 'recall', 'f1']).T

def log_tensorboard(board, losses, labels, preds, mappings, epoch, mode='Train'):

    loss = losses['loss']
    age_loss = losses['age']
    race_loss = losses['race']
    gender_loss = losses['gender']

    age_labels = labels['age']
    race_labels = labels['race']
    gender_labels = labels['gender']

    age_preds = preds['age']
    race_preds = preds['race']
    gender_preds = preds['gender']

    accuracy_str = f'Age accuracy = {accuracy_score(age_labels, age_preds)}\t'
    accuracy_str += f'Race accuracy = {accuracy_score(race_labels, race_preds)}\t'
    accuracy_str += f'Gender accuracy = {accuracy_score(gender_labels, gender_preds)}'

    print(f'Epoch = {epoch}, Loss = {loss}')
    print(accuracy_str)
    print(f'Age loss = {age_loss}\tRace loss = {race_loss}\tGender Loss = {gender_loss}')
    
    board.add_scalar(f'{mode} Loss', loss, epoch)
    board.add_scalar(f'Age {mode} Loss', age_loss, epoch)
    board.add_scalar(f'Race {mode} Loss', race_loss, epoch)
    board.add_scalar(f'Gender {mode} Loss', gender_loss, epoch)

    board.add_scalar(f'Age {mode} accuracy', accuracy_score(age_labels, age_preds), epoch)
    board.add_scalar(f'Race {mode} accuracy', accuracy_score(race_labels, race_preds), epoch)
    board.add_scalar(f'Gender {mode} accuracy', accuracy_score(gender_labels, gender_preds), epoch)

    board.add_scalar(f'Age {mode} f1', f1_score(age_labels, age_preds, average='macro'), epoch)
    board.add_scalar(f'Race {mode} f1', f1_score(race_labels, race_preds, average='macro'), epoch)
    board.add_scalar(f'Gender {mode} f1', f1_score(gender_labels, gender_preds, average='macro'), epoch)

    board.add_scalar(f'Age {mode} precision', precision_score(age_labels, age_preds, average='macro'), epoch)
    board.add_scalar(f'Race {mode} precision', precision_score(race_labels, race_preds, average='macro'), epoch)
    board.add_scalar(f'Gender {mode} precision', precision_score(gender_labels, gender_preds, average='macro'), epoch)

    board.add_scalar(f'Age {mode} recall', recall_score(age_labels, age_preds, average='macro'), epoch)
    board.add_scalar(f'Race {mode} recall', recall_score(race_labels, race_preds, average='macro'), epoch)
    board.add_scalar(f'Gender {mode} recall', recall_score(gender_labels, gender_preds, average='macro'), epoch)

    # Create confusion matrix each epoch and label it the same way for more ease comparing during epochs
    board.add_figure(f'{mode.lower()}_fairface_cm', create_cfs(ages={'true' : age_labels, 'pred' : age_preds}, 
                        genders={'true' : gender_labels, 'pred' : gender_preds},
                        races={'true' :   race_labels, 'pred' : race_preds}, 
                        mappings=mappings), global_step=epoch)
    
    return board