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
from sklearn.svm import LinearSVC


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

	clf = LinearSVC(random_state=0)

	clf.fit(X_train, Y_train)
	score = clf.score(X_test, Y_test)

	scores.append(score)

endTime = time.time()
	
with open('results/' + name + '_svm_' + type + '.txt', 'w') as file:
	file.write('Score: ' + str(np.average(scores)) + '\n')
	file.write('Time: ' + str(endTime - startTime))
	file.close()

print('Score: ' + str(np.average(scores)))
print('Time: ' + str(endTime - startTime))
