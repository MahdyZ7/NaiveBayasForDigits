# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict



# multi class classification
class NaiveBayes:
	def __init__(self):
		self.classes = None
		self.class_freq = None
		self.prob = None

	def fit(self, X, y):
		self.classes, class_freq = np.unique(y, return_counts=True)
		self.class_freq = dict(zip(self.classes, class_freq))
		self.prob = self._calculate_probability(X, y)

	def _calculate_probability(self, X, y):
		prob = defaultdict(dict)
		for c in self.classes:
			X_c = X[y == c]
			for i in range(X.shape[1]):
				values, counts = np.unique(X_c[:, i], return_counts=True)
				prob[c][i] = dict(zip(values, counts))
		return prob

	def _calculate_posterior(self, X):
		posteriors = []
		for c in self.classes:
			class_prior = np.log(self.class_freq[c] / sum(self.class_freq.values()))
			posterior = np.sum([np.log((self.prob[c][i][val] + 1) / (sum(self.prob[c][i].values()) + len(self.prob[c][i]))) 
								if val in self.prob[c][i] else np.log(1 / (sum(self.prob[c][i].values()) + len(self.prob[c][i]))) 
								for i, val in enumerate(X)])
			posterior = class_prior + posterior
			posteriors.append(posterior)
		return self.classes[np.argmax(posteriors)]

	def predict(self, X):
		return np.array([self._calculate_posterior(x) for x in X])

	def report(self, target, output, string):
		print(string, "\n\n")
		print(metrics.classification_report(target, output, digits=2))
		print(metrics.confusion_matrix(target, output))
		print("Overall Accuracy: ", "%.2f" % metrics.accuracy_score(target, output))
		print("Overall precision score: ", "%.2f" % metrics.precision_score(target, output, average='weighted'))
		print("Overall Recall score: ", "%.2f" % metrics.recall_score(target, output, average='weighted'))
		print("Overall F1 score: ", "%.2f" % metrics.f1_score(target, output, average='weighted'))
		print("\n\n")


# Main program
	# define consants
RANDOM_SEED = 41
SPLIT_RATIO = 0.3
	# load data
digits = datasets.load_digits()
	#print some data info
print("Data set headers: ", digits.keys())
print(digits.DESCR)
print("Size of data", digits.data.shape)



	# iniitialize classifier, data and target
clf = NaiveBayes()
data = digits.data
target = digits.target
	# shuffel data into training and testing
data, target = shuffle(data, target, random_state = RANDOM_SEED)
	# split data into training and testing
train_data, test_data = train_test_split(data, test_size=SPLIT_RATIO, shuffle=False)
train_target, test_target = train_test_split(target, test_size=SPLIT_RATIO, shuffle=False)
print("The training data size is: ", train_data.shape,"The testing data size is: ", test_data.shape)

# train the data
clf.fit(train_data, train_target)
# test on training data
train_prediction = clf.predict(train_data)
clf.report(train_target, train_prediction, "Training Data")
# pridict the data
predicted = clf.predict(test_data)
clf = clf.report(test_target, predicted, "Testing Data")

# print("Accuracy: ", metrics.accuracy_score(test_target, predicted))
# print("Precision: ", metrics.precision_score(test_target, predicted, average='weighted'))
# print("Recall: ", metrics.recall_score(test_target, predicted, average='weighted'))
# print("F1 score: ", metrics.f1_score(test_target, predicted, average='weighted'))

# print(metrics.classification_report(test_target, predicted, digits=2))
# print(metrics.confusion_matrix(test_target, predicted))
