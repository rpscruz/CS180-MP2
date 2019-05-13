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
import os

from sklearn.neural_network import MLPClassifier
 
print("[INFO] loading scan attributes...")
# ave_depth = get_ave_depth()
# memoize: 186.
ave_depth = 186
scans, masks = datasets.load_scans(3, ave_depth)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing

print("[INFO] processing data...")
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
split = train_test_split(scans, masks, test_size=0.25, random_state=42)

print(len(scans))
print(len(masks))
print("\n")


(X_train, X_test, y_train, y_test) = split

print("training sets")
print(len(X_train))
print(len(X_train[0]))
print(len(X_train[0][0]))
print(len(X_train[0][0][0]))
print(len(y_train[0][0]))

print("testing sets")
print(len(X_test))
print(len(y_test))

num_classes = 5
'''
The second is our soft-max classification, or output layer, which is the size of the number of our classes (10 in this case, for our 10 possible hand-written digits).
'''

from keras.models import Sequential, Input
from keras.layers import Dense, Conv2D, Conv1D, Flatten, MaxPooling2D

model = Sequential()
print(model.summary())
'''
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(None, 512, 512)))
'''




print("shape")
print(X_train.shape)

sample, row, col = X_train[0].shape
print("sample", sample)
print("row", row)
print("col", col)
# 4D input.
x = Input(shape=(sample, row, col))

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

# Encodes a row of pixels using TimeDistributed Wrapper.
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# Encodes columns of encoded rows.
encoded_columns = LSTM(col_hidden)(encoded_rows)

# Final predictions and model.
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 10
epochs = 10

# Training.

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)
X_test = np.vstack(X_test)
y_test = np.vstack(y_test)


model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

'''
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
