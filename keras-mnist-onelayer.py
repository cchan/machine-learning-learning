import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32')/255
X_test = X_test.reshape(10000, 784).astype('float32')/255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = Sequential()
model.add(Dense(output_dim=10, input_dim=784))
model.add(Activation("softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.fit(X_train, Y_train, nb_epoch=10, batch_size=100)

score = model.evaluate(X_test, Y_test, verbose=0)
print('Score', score[0])
print('Accuracy', score[1])
