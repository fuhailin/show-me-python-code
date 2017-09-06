'''Trains a simple convnet on the kaggle Digit Recognizer dataset.
Gets to 98.81% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
7 seconds per epoch on a GeForce GTX 1050 GPU.
'''
from __future__ import print_function
import time
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split

batch_size = 128
num_classes = 10
epochs = 12
img_rows, img_cols = 28, 28  # input image dimensions
# the data, shuffled and split between train and test sets
# create the training & test sets, skipping the header row with [1:]
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
x_train = train.ix[:, 1:].values.astype('float32')  # all pixel values
y_train = train.ix[:, 0].values.astype('int32')  # only labels i.e targets digits
x_test = test.values.astype('float32')

y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)  # convert class vectors to binary class matrices
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)
# X_train = x_train.reshape(x_train.shape[0], 28, 28)# Convert train datset to (num_images, img_rows, img_cols) format

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)  # convert class vectors to binary class matrices expand 1 more dimention as 1 for colour channel gray
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    X_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    X_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train /= 255
x_val /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_val.shape[0], 'test samples')


def init_model():
    start_time = time.time()
    print('Compiling Model ... ')
    model = Sequential()
    model.add(Conv2D(32, activation='relu',
                     input_shape=input_shape,
                     nb_row=3, nb_col=3))
    model.add(Conv2D(64, activation='relu',
                     nb_row=3, nb_col=3))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('Model compield in {0} seconds'.format(time.time() - start_time))
    return model


model = init_model()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:{0}    Test accuracy:{1}', score[0], score[1])
predictions = model.predict_classes(X_test, verbose=0)
submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions) + 1)), "Label": predictions})
submissions.to_csv("../data/submissions_cnn.csv", index=False, header=True)
