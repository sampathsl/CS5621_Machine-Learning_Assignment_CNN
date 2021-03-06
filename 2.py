import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

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

# define the mode
model = Sequential()

# input_shape
input_shape = (28, 28, 1)

# first 32 neuron layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))

# define the max pool
model.add(MaxPooling2D())

# flatten the images
model.add(Flatten())

# add a dense layer
model.add(Dense(128, activation='relu'))

# add a dense layer
num_classes = 10
model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# define the noise factor
noise_factor = 0.05
X_train = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train = np.clip(X_train, 0., 1.)
X_test = np.clip(X_test, 0., 1.)

# train the model with train data
model.fit(X_train, Y_train, epochs=20, batch_size=200, validation_data=(X_test, Y_test), verbose=1)

# test the model with test data
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

print('Loss for the CNN model = %.3f' % loss)
print('Accuracy for the CNN model = %.3f' % (accuracy * 100.0))