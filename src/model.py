import tensorflow as tf
from keras import layers, models
def build_model(input_shape):

    model = models.Sequential()

    model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D( pool_size = (2, 2)))

    model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D( pool_size = (2, 2)))

    model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D( pool_size = (2, 2)))
    return model