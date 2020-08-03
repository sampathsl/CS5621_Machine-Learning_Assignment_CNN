from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# load the data set
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# reshape the train dataset to have a single channel
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# reshape the test dataset to have a single channel
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# normalize the pixel values from a scale out of 255 to a scale out of 1
# convert from integers to floats
X_train = X_train.astype('float32')
X_train /= 255

# convert from integers to floats
X_test = X_test.astype('float32')
X_test /= 255

# convert Y_train to categorical
Y_train = to_categorical(Y_train, 10)

# convert Y_test to categorical
Y_test = to_categorical(Y_test, 10)

# create sequential model
model = Sequential()

# update the model with additional layers with different parameters
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(512, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# implement the data augmentation to prevent over fitting
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False)

# defined fixed batch size
batch_size = 64
train_gen = datagen.flow(X_train, Y_train, batch_size=batch_size)
test_gen = datagen.flow(X_test, Y_test, batch_size=batch_size)

epochs = 20
history = model.fit_generator(train_gen,
                              epochs=epochs,
                              steps_per_epoch=X_train.shape[0],
                              validation_data=test_gen,
                              validation_steps=X_test.shape[0])
