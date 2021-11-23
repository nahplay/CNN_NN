import tensorflow as tf
from ImageClassificationTF import ImageClassificationTF
from ImageClassificationPT import ImageClassificationPT


# GPU usage
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Tensorflow/Keras
def tensorflow_classifier():
    image_class_keras = ImageClassificationTF('hotdog__not_hotdog', image_size=(300, 300))  # Fitting samples into a class
    train_val_test = image_class_keras.test_train_val()  # Data split
    image_class_keras.visualization()
    image_class_keras.augumented_visualization()  # Augmented data visualization
    own_model = image_class_keras.customized_model((300, 300),
                                                   2)  # Own model, picture size and number of classes set as parameters
    resnet_f = image_class_keras.pretrained_model(model='resnet', unfreeze=0)  # Pretrained resnet with freezed layers
    vgg19_f = image_class_keras.pretrained_model(model='VGG19', unfreeze=0)  # Pretrained VGG19 with freezed layers
    resnet_u = image_class_keras.pretrained_model(model='resnet', unfreeze=1)  # Pretrained resnet with unfreezed layers
    vgg19_u = image_class_keras.pretrained_model(model='VGG19', unfreeze=1)  # Pretrained VGG19 with unfreezed layers
    #image_class_keras.training_process(own_model, 50, *train_val_test)
    models = [own_model, resnet_f, vgg19_f, resnet_u, vgg19_u]

    for model in models:
        image_class_keras.training_process(model, 100, *train_val_test)


# Pytorch
def torch_classifier():
    imageclass = ImageClassificationPT('hotdog__not_hotdog', (315, 315))
    train_val_test = imageclass.load_split_train_test(0.2)
    imageclass.visualize_classification(train_val_test[0])

    # Resnet50 with frozen layers
    imageclass.pretrained_model('resnet', 0, *imageclass.load_split_train_test(0.2), 50)
    # Densenet with frozen layers
    imageclass.pretrained_model('densenet', 0, *imageclass.load_split_train_test(0.2), 50)
    # Resnet50 with unfreezed layers
    imageclass.pretrained_model('resnet', 1, *imageclass.load_split_train_test(0.2), 50)
    # Densenet with unfreezed layers
    imageclass.pretrained_model('densenet', 1, *imageclass.load_split_train_test(0.2), 50)


if __name__ == "__main__":
    tensorflow_classifier()
    torch_classifier()
