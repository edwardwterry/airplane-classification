from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import random
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

#Import wandb libraries
import wandb
from wandb.keras import WandbCallback

#Initialize wandb
wandb.init(project="airplane-classification")
config = wandb.config

#Track hyperparameters
config.dropout = 0.2
config.hidden_layer_size = 128
config.layer_1_size  = 16
config.layer_2_size = 32
config.learn_rate = 0.01
config.decay = 1e-6
config.momentum = 0.9
config.epochs = 20

# load files in
img_width=128
img_height=128

files = os.listdir(os.getcwd() + '/images')
# print (files)
X = []
y = []
labels =["737", "777", "787"]

for f in files:
    path = os.getcwd() + '/images/' + f
    category = f.split('_')[0]
    img = cv2.imread(path, 0) # 0 grayscale
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype('float32')
    img /= 255.
    X.append(img)
    y.append(labels.index(category))

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Reducing the dataset size to 10,000 examples for faster train time
# true = list(map(lambda x: True if random.random() < 0.167 else False, range(60000)))
# ind = []
# for i, x in enumerate(true):
#     if x == True: ind.append(i)

# X_train = X_train_orig[ind, :, :]
# y_train = y_train_orig[ind]


# #reshape input data
X_train = X_train.reshape(X_train.shape[0], img_width, img_height, 1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height, 1)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

sgd = SGD(lr=config.learn_rate, decay=config.decay, momentum=config.momentum,
                            nesterov=True)

# build model
model = Sequential()
model.add(Conv2D(config.layer_1_size, (5, 5), activation='relu',
                            input_shape=(img_width, img_height,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(config.layer_2_size, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(config.dropout))
model.add(Flatten())
model.add(Dense(config.hidden_layer_size, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

#Add Keras WandbCallback
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train, y_train,  validation_data=(X_test, y_test), epochs=config.epochs,
    callbacks=[WandbCallback(data_type="image", labels=labels)])
