"""
To know more or get code samples, please visit my website:
https://morvanzhou.github.io/tutorials/
Or search: 莫烦Python
Thank you for supporting!
"""

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 8 - RNN Classifier example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

TIME_STEPS = 28  # same as the height of the image
INPUT_SIZE = 28  # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50
LR = 0.001
num_classes = 10

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
x_train = train.ix[:, 1:].values.astype('float32')  # all pixel values
y_train = train.ix[:, 0].values.astype('int32')  # only labels i.e targets digits
x_test = test.values.astype('float32')

y_train = to_categorical(y_train, num_classes)  # convert class vectors to binary class matrices
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

# data pre-processing
X_train = x_train.reshape(-1, 28, 28) / 255.  # normalize
X_val = x_val.reshape(-1, 28, 28) / 255.  # normalize
X_test = x_test.reshape(-1, 28, 28) / 255.  # normalize

# build RNN model
model = Sequential()
# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),  # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))
# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
for step in range(4001):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
    Y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
    cost = model.train_on_batch(X_batch, Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost, accuracy = model.evaluate(X_val, y_val, batch_size=y_val.shape[0], verbose=False)
        print('test cost: ', cost, 'test accuracy: ', accuracy)
predictions = model.predict_classes(X_test, verbose=0)
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})
submissions.to_csv("../data/submissions_rnn.csv", index=False, header=True)
