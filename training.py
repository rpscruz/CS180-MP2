# import the necessary packages
import datasets
import models
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier

print("main: processing data...")
# ave_depth = get_ave_depth()
# memoize: 186.
ave_depth = 186
scans, masks = datasets.load_scans(5, ave_depth)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
print("[INFO] processing data...")
split = train_test_split(scans, masks, test_size=0.25, random_state=42)
(X_train, X_test, y_train, y_test) = split

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(X_train.shape[1:])
print(y_train.shape[1:])
print(X_test.shape[1:])
print(y_test.shape[1:])
slide, row, col = X_train.shape[1:]


from keras.models import Sequential
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
from keras.layers import Dense, Conv3D, Conv2D, Conv1D, Flatten, MaxPooling2D


# Training parameters
batch_size = 100 # original = 32
num_classes = 10
epochs = 30

print("main: creating model")
model = Sequential()
model.add(Conv3D(32,kernel_size = 2, input_shape = (512, 512, 186)))

# model.add(Flatten())
print("main: compiling model")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print("main: fitting model")
history = model.fit(X_train, y_train, 
          batch_size = batch_size,
          epochs=epochs, 
          verbose=1, 
          validation_data=(X_test,y_test))

# Evaluation
scores = model.evaluate(X_test, y_test, verbose = 1)
print('Test loss', scores[0])
print('Test accuracy', scores[1])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, acc, 'g', label='Validation accuracy')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.title('Training and validation accuracy')
plt.legend()

fig_acc = plt.figure()
fig_acc.savefig('accuracy.png')

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()
fig_loss = plt.figure()
fig_loss.savefig('loss.png')
plt.show()


'''

model = Sequential()
print(model.summary())
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(None, 512, 512)))



# Training.

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

model.add(Dense(256, activation='relu'))

print(model.summary())

from keras.losses import categorical_crossentropy
from keras import optimizers

model.compile(loss=categorical_crossentropy,
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['accuracy'])


from keras.callbacks import Callback
class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
history = AccuracyHistory()

model.fit(X_train, y_train,
          batch_size=4,
          epochs=10, # epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[history])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



'''
