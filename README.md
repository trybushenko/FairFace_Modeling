# FairFace Classification

This project aims to classify faces into different categories such as age, gender, and race, using the FairFace dataset. The model used is a MobileNet V3 Large pretrained on ImageNet dataset, and the last layer is modified and adapted to the current FairFace dataset.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

To install the required packages, run the following command:

```bash
sh runthis.sh
```

## Usage

To use the model for inference, run the following command:

```bash
python inference.py
```

This will load the checkpoint file and run inference on the test dataset. The results will be saved to a CSV file.

To use the model for training, run the following command:

```bash
python train.py
```

This will start the training process using the FairFace dataset. The training process will be logged to TensorBoard, and the best model will be saved to a checkpoint file.

## Training

The training process consists of the following steps:

1. Load the FairFace dataset and create the dataloaders.
2. Initialize the model and move it to the GPU.
3. Define the loss function and optimizer.
4. Train the model for a given number of epochs.
5. Evaluate the model on the test dataset.
6. Save the best model checkpoint.

## Results

The results of the training process can be viewed using TensorBoard. To start TensorBoard, run the following command:

```bash
tensorboard --logdir runs
```

Then open your web browser and go to http://localhost:6006.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

I would like to thank the FairFace team for making their dataset available, and the PyTorch team for making their library so easy to use. I would also like to thank my mentors and classmates for their feedback and support during the development of this project.