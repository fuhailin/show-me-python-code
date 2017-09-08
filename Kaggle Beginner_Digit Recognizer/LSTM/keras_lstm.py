from __future__ import print_function

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
import sys, os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

num_classes = 10
nb_epochs = 1000

batch_size = 100  # 100
hidden_units = 50  # 50

# rmsprop
learning_rate = 0.01  # 0.001
rho = 0.9  # 0.9

clip_norm = 5.0  # 5.0
forget_bias = 1.0

train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
x_train = train.ix[:, 1:].values.astype('float32')  # all pixel values
y_train = train.ix[:, 0].values.astype('int32')  # only labels i.e targets digits
x_test = test.values.astype('float32')

y_train = to_categorical(y_train, num_classes)  # convert class vectors to binary class matrices
x_train, x_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

X_train = x_train.reshape(x_train.shape[0], -1, 1)
X_val = x_val.reshape(x_val.shape[0], -1, 1)
X_test = x_test.reshape(x_test.shape[0], -1, 1)

X_means = np.mean(X_train, axis=0)
X_stds = np.std(X_train, axis=0)
X_train = (X_train - X_means) / (X_stds + 1e-6)
X_val = (X_val - X_means) / (X_stds + 1e-6)
X_test = (X_test - X_means) / (X_stds + 1e-6)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_val.shape[0], 'val samples')

# lambda shape: forget_bias*one(shape, name=None)

print('Compare to LSTM...')
model = Sequential()
model.add(LSTM(hidden_units, input_shape=X_train.shape[1:], inner_init='glorot_uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(clipnorm=clip_norm)
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

checkpointer = ModelCheckpoint(
    filepath="../data/rnn-mist/lstm-weights" + "-bs-" + str(batch_size) + "-hu-" + str(hidden_units) + "-lr-" + str(learning_rate) + "-rho-" + str(rho) + "-clip-" + str(
        clip_norm) + "-epoch-{epoch:02d}-val-{val_acc:.2f}" + ".hdf5", verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_acc', patience=15, verbose=1, mode='max')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(X_val, Y_val),
          callbacks=[earlystopper, checkpointer]
          )

scores = model.evaluate(X_val, Y_val, verbose=0)
print('LSTM test score:', scores[0])
print('LSTM test accuracy:', scores[1])

predictions = model.predict_classes(X_test, verbose=0)
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})
submissions.to_csv("../data/submissions_lstm.csv", index=False, header=True)
