import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold, StratifiedKFold
from sklearn import preprocessing
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Activation
from keras.optimizers import Adam, RMSprop, SGD, Adamax


# filenames = ['alon', 'borovecki', 'burczynski', 'chiaretti', 'chin', 'chowdary', 'christensen', 'golub', 'gordon', 'gravier', 'khan', 'nakayama', 'pomeroy', 'shipp', 'singh', 'sorlie', 'su', 'subramanian', 'sun', 'tian', 'west', 'yeoh']

name = sys.argv[1]
type = sys.argv[2]

features = pd.read_csv('data/' + name + '_inputs.csv', header = None)
labels = pd.read_csv('data/' + name + '_outputs.csv', header = None)

features.fillna(0, inplace = True)

features = np.asarray(features.values)
labels = np.transpose(np.asarray(labels.values.ravel() - 1, dtype=int))

min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

if type == 'lasso':
	gains = np.asarray(np.loadtxt('features/' + name + '_lasso.txt'))
	indexes = np.where(gains != 0)[0]
else:
	gains = np.asarray(np.loadtxt('features/' + name + '_lasso.txt'))
	indexes = np.where(gains != 0)[0]
	gains = np.asarray(np.loadtxt('features/' + name + '_relieff.txt')) 
	indexes = gains.argsort()[-indexes.shape[0]:][::-1]


scores = []

loo = LeaveOneOut()

startTime = time.time()

for train_index, test_index in loo.split(features):
	x_train, x_test = features[train_index], features[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
	
	X_train = x_train[:, indexes]
	X_test = x_test[:, indexes]
	Y_train = y_train[:]
	Y_test = y_test[:]
	
	batch_size = 1
	num_classes = np.max(labels) + 1
	epochs = 50
	
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	Y_train = Y_train[:]
	Y_test = Y_test[:]
	# print(X_train.shape[0], 'train samples, ', Y_train.shape)
	# print(X_test.shape[0], 'test samples, ', Y_test.shape)
	
	# convert class vectors to binary class matrices
	Y_train = keras.utils.to_categorical(Y_train, num_classes)
	Y_test = keras.utils.to_categorical(Y_test, num_classes)
	
	model = Sequential()
	
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(200, input_dim=X_train.shape[1], kernel_initializer='lecun_uniform', activation='relu'))
	model.add(Dense(100, kernel_initializer='lecun_uniform', activation='relu'))
	model.add(Dense(Y_train.shape[1], kernel_initializer='lecun_uniform', activation='softmax'))
	
	sgd = SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=Adamax(), metrics=['accuracy'])
	
	# model.summary()
	
	history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data=(X_test, Y_test))

	score = model.evaluate(X_test, Y_test, verbose=0)
	
	scores.append(score[1])

endTime = time.time()
	
with open('results/' + name + '_mlp_' + type + '.txt', 'w') as file:
	file.write('Score: ' + str(np.average(scores)) + '\n')
	file.write('Time: ' + str(endTime - startTime))
	file.close()

print('Score: ' + str(np.average(scores)))
print('Time: ' + str(endTime - startTime))
