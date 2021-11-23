Here is the implementation of Resnet, VGG19 - with Keras and
Resnet50 and Densenet161 with Pytorch. Layers of base model were set to non trainable and then set as trainable to compare the performance of the models.
There are some visualization methods as well. Samples visualization was performed only with keras, to omit repetitions, yet augmentation visualization was performed using both keras and torch libraries.
Optimal learning rate for all models was implemented to determine visually and to be set on user choice.(Leslie N. Smith method - https://arxiv.org/abs/1506.01186)
All visualizations will be stored to the corresponding folder(*project_folder/results/**library)


*location of all of the files for the project
**library which you are using - keras/pytorch

