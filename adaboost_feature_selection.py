import math
import numpy as np
import adaboost
from sklearn.datasets import load_digits

#### Steps:
# 1. find optimal classifiers for each feature
# 2. run Adaboost on those classifiers and return what they pick

digits = load_digits(n_class=2)
X,y = digits.data,digits.target

print 'X',X.shape
print 'X',X
# print X[:,5]
# print 'y',sum(y)


def test_feature(feature,X,y):
	error = 0.0
	for i in range(len(y)):
		tmp = False
		if y[i] == 1:
			tmp = True

		if feature(X[i]) != y[i]:
			error += 1

	return error


def digits_make_classifiers(X,y):
	output = []
	for num_feature in range(X.shape[1]):
	# for num_feature in range(1):
		# print 'num_feature',num_feature
		# print 'min',int(min(X[:,num_feature]))
		# print 'max',int(max(X[:,num_feature]))
		best_error = float('inf')
		best_cutoff = float('inf')
		best_feature = lambda x: 1
		for cutoff in range(int(min(X[:,num_feature])),int(max(X[:,num_feature]))):
			# Test >=
			feature = lambda x: x[num_feature] >= cutoff
			error = test_feature(feature,X,y)
			# print 'error',error
			if error < best_error:
				best_error = error
				best_cutoff = cutoff
				best_feature = feature
			# Test <=
			feature = lambda x: x[num_feature] <= cutoff
			error = test_feature(feature,X,y)
			# print 'error',error
			if error < best_error:
				best_error = error
				best_cutoff = cutoff
				best_feature = feature

		output.append((num_feature,best_cutoff,best_feature))


	return output

def digits_make_classifiers_to_misclassified(X,Y,classifiers,ids_to_points):
	"""
	Takes a list of classifiers and returns the a dictionary mapping classifiers to the points they misclassify
	
	This method is specific to the format of the classifiers
	"""
	output = {key: [] for key in classifiers}
	N = len(X)
	for cf in classifiers:
		for i in range(N):
			cf_classification = cf[2](X[i])
			if cf_classification != Y[i]:
				# output[cf].append(X[i])
				output[cf].append(adaboost.key_from_value(ids_to_points,X[i]))

	return output


digits_classifiers = digits_make_classifiers(X,y)
digits_ids_to_points = adaboost.make_point_identifiers(X)
digits_classifiers_to_misclassified = digits_make_classifiers_to_misclassified(X,y,digits_classifiers,digits_ids_to_points)

digits_points = digits_ids_to_points.keys()
resulting_classifier = adaboost.adaboost(digits_points,digits_classifiers_to_misclassified,max_num_rounds=20)
print 'resulting_classifier',resulting_classifier
features_chosen = [i[0][0] for i in resulting_classifier]
print 'features_chosen',features_chosen

# Testing
# feature_test = lambda x,cutoff: x[0] > cutoff
# print test_feature(feature_test,X,y,0)
# print sum(y)