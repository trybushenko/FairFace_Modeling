import numpy as np
import pandas as pd
import torch
import shutil
import random

def invert_weights(weights):
    """Inverts a list of weights and normalizes them so that they sum to 1.

    Args:
        weights: A list of weights.

    Returns:
        A list of normalized weights.
    """

    weights_inv = [1 / i for i in weights]
    weights_sum_norm = sum(weights_inv) / len(weights_inv)
    return [i / weights_sum_norm for i in weights_inv]


def calc_tasks_weight(class_maps: dict):
    """Calculates the weights for a set of tasks, where the weights are inversely proportional to the number of subcategories in each task.

    Args:
        class_maps: A dictionary of class maps, where each class map is a dictionary that maps class names to lists of subcategories.

    Returns:
        A dictionary of class weights, where each class weight is a float.
    """

    out_name = ['age', 'race', 'gender']
    out_len = [len(class_maps[out_class]) for out_class in out_name]
    total_out = sum(out_len)

    weights = [total_out / i for i in out_len]
    class_weights = invert_weights(weights)
    class_weights = list(map(lambda x: round(x, 2), class_weights))

    # the more subcategories in a super category, the less weight it has
    for k, v in zip(out_name, class_weights):
        print(f"{k:>10}: {v:.2f}")

    return dict(zip(out_name, class_weights))

def calc_category_weights(train_samples:np.array, train_test_labels:pd.DataFrame, dicts: dict):
    """Calculates the category weights for the given train samples and train-test labels.

    Args:
        train_samples: A NumPy array of train samples.
        train_test_labels: A Pandas DataFrame of train-test labels.
        dicts: A dictionary of mapping dictionaries for the age, gender, and race categories.

    Returns:
        A Pandas DataFrame of train-test labels with the category weights added.
    """

    # Extract the age, gender, and race mapping dictionaries from the `dicts` dictionary.
    age_dict, race_dict, gender_dict = (dicts[i] for i in ('age_dict', 'race_dict', 'gender_dict'))

    # Create a Pandas DataFrame from the train samples.
    train_samples = pd.DataFrame({'file' : train_samples})

    # Merge the train samples DataFrame with the train-test labels DataFrame on the `file` column.
    train_test_labels = train_test_labels.merge(train_samples, on='file', how='right')

    # Drop the `service_test` column from the train-test labels DataFrame.
    train_test_labels = train_test_labels.drop(['service_test'], axis=1)

    # Create a list of the category labels.
    seq = ['age', 'gender', 'race']

    # Create a list of value counts DataFrames for each category.
    vcs = [train_test_labels[label].value_counts().to_frame().reset_index() for label in seq]

    # Create a list of mapping dictionaries for each category.
    dicts = [age_dict, gender_dict, race_dict]

    # Transform the value counts DataFrames using the mapping dictionaries.
    transform = [vc[label].map(mapping) for (vc, label, mapping) in zip(vcs, seq, dicts)]

    # Apply the transformed value counts to the original dataframes.
    for i in range(len(seq)):
        vcs[i][seq[i]] = transform[i]

    # Sort the value counts DataFrames by category.
    vcs = [vcs[i].sort_values(seq[i]) for i in range(len(seq))]

    # Calculate the class weights for each category.
    for i in range(len(seq)):
        vcs[i]['class_weight'] = 1 / vcs[i]['count']

    # Simultaneously apply the mapping dictionaries to the train-test labels DataFrame and create a new column for the class weights.
    for i in range(len(seq)):
        train_test_labels[seq[i]] = train_test_labels[seq[i]].map(dicts[i])
        train_test_labels[f'{seq[i]}_weight'] = train_test_labels[seq[i]].apply(lambda x: vcs[i]['class_weight'].values[x])

    # Calculate the total weight for each sample.
    train_test_labels['total_weight'] = train_test_labels[[f'{label}_weight' for label in seq]].apply(np.sum, axis=1)

    # Extract the file number from the file name.
    train_test_labels['file_num'] = train_test_labels['file'].str.extract('(\d+)', expand=False).astype(int)

    # Sort the train-test labels DataFrame by file number.
    train_test_labels = train_test_labels.sort_values('file_num')

    # Drop the file number column from the train-test labels DataFrame.
    train_test_labels = train_test_labels.drop(['file_num'], axis=1)

    return train_test_labels

def checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    """
    Checkpoints a model.

    Args:
        state: The model state to be checkpointed.
        is_best: Whether model is best or not
        checkpoint_dir: The directory of the checkpoint.
        best_model_dir: The directory of the best model.

    Returns:
        None
    """
    f_path = checkpoint_dir / 'checkpoint.pt'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / 'best_model.pt'
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """Loads a checkpoint file.

    Args:
        checkpoint_fpath: Path to the checkpoint file.
        model: The model to load the checkpoint into.
        optimizer: The optimizer to load the checkpoint into.

    Returns:
        The model, optimizer, and epoch number.
    """

    # Load the checkpoint file.
    checkpoint = torch.load(checkpoint_fpath)

    # Load the model's state dictionary from the checkpoint.
    model.load_state_dict(checkpoint['state_dict'])

    # Load the optimizer's state dictionary from the checkpoint.
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Return the loaded model, optimizer, and epoch number.
    return model, optimizer, checkpoint['epoch']

def get_mappings(class_dict):
    """
    Returns a list of the sorted keys of a dictionary.

    Args:
        class_dict: A dictionary mapping labels to their corresponding indices.

    Returns:
        A list of the sorted keys of the dictionary.
    """
    return [i[0] for i in sorted(list(class_dict.items()), key=lambda x: x[1])]

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

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set torch.backends.cudnn.deterministic
            to True and torch.backends.cudnn.benchmark to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False