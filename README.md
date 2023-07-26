In this project, I explored the FairFace dataset with Exploratory Data Analysis, and Images Observance and Currently trying to get state-of-the-art result with MobileNetV3 Large model. The last step of the previous sentence includes fine-tuning, modeling, image augmentation, and processing. 
Next steps to apply:
1) img-aug library;
2) albumentation;
3) Pytorch-lightning

I tracked model training and the next performance with Tensorboard logging necessary metrics and losses to the directory, which wasn't pushed here because of its big size.

Here is the architecture of the last layer of the fine-tuned model, which was previously pre-trained on the ImageNet dataset (popular benchmark dataset). 
![alt text](https://github.com/trybushenko/FairFace_Modeling/blob/master/model_architecture.png)
**To launch a project run ```pip install -e .``` in the root of the project.**
