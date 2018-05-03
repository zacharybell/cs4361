from __future__ import print_function

from scipy.special import expit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten

np.set_printoptions(threshold=np.inf) #Print complete arrays
batch_size = 64
epochs = 100
learning_rate = 0.0001
first_layer_filters = 32
second_layer_filters = 64
ks = 4
mp = 2
dense_layer_size = 128



model = Sequential()
model.add(Conv2D(first_layer_filters, kernel_size=(ks, ks),
                 activation='relu',
                 input_shape= (64, 64, 1)))
model.add(MaxPooling2D(pool_size=(mp, mp)))
model.add(Conv2D(second_layer_filters, (ks, ks), activation='relu'))
model.add(MaxPooling2D(pool_size=(mp, mp)))
model.add(Flatten())
model.add(Dense(dense_layer_size, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

#Train network, store history in history variable
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1)

score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

pred=model.predict_classes(x_test)
print('\n',confusion_matrix(test_class,pred))
print(model.summary())
