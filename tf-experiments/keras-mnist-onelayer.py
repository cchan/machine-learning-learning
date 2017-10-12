import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

# What shape is it in originally??
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Shape it to a nice easy-to-use thing
X_train = X_train.reshape(60000, 784).astype('float32')/255
X_test = X_test.reshape(10000, 784).astype('float32')/255

# Make it one-hot
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Dimensions
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# Simple softmaxed linear regression
model = Sequential()
model.add(Dense(output_dim=10, input_dim=784))
model.add(Activation("softmax"))

# Nice human readable summary
model.summary()

# This actually puts it together
# idk what categorical cross-entropy is, sgd is stochastic gradient descent (which means it's online, as opposed to batch gradient descent)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# This actually trains it on the training set
model.fit(X_train, Y_train, nb_epoch=10, batch_size=100) # nb_epoch: number of epochs (number of repeats of the whole set? or just of batch_size?)

# Evaluate it in accuracy on the test set
score = model.evaluate(X_test, Y_test, verbose=0)

print('Score', score[0])
print('Accuracy', score[1])
