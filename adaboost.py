import math
import numpy as np

# TODO could put use_smallest_error thing (also goes in choose_best_classifier function)


# Dummy dataset -> see paper for demo http://courses.csail.mit.edu/6.034f/Examinations/2015s4.pdf (A1)
test_X = np.array([[0.5,2.5],[1.5,1.5],[2.5,2.5],[3.5,2.5]]) # A, B, C, D (0,1,2,3)
test_Y = np.array([1,1,1,1])
test_classifiers = [('h1',lambda x: x[0] >= 1),('h2',lambda x: x[1] >= 2),('h3',lambda x: x[0] <= 2),
('h4',lambda x: x[0] >= 3)]

def make_point_identifiers(X):
	"""
	Numpy arrays are not hashable, so this helper function makes a dictionary of integers to points,
	so that the integers can be used to keep track of points in the adaboost algorithm.
	"""
	output = {}
	for i in range(len(X)):
		output[i] = X[i]
	return output

test_ids_to_points = make_point_identifiers(test_X)

def key_from_value(d,target_value):
	"""
	Helper function to return key from a given value in a dictionary
	"""
	for key, value in d.items():
		# if value == target_value:
		if np.array_equal(value,target_value):
			return key

	print "key not found"
	return None


def test_make_classifiers_to_misclassified(X,Y,classifiers,ids_to_points):
	"""
	Takes a list of classifiers and returns the a dictionary mapping classifiers to the points they misclassify
	
	This method is specific to the format of the classifiers
	"""
	output = {key: [] for key in classifiers}
	N = len(X)
	for cf in classifiers:
		for i in range(N):
			cf_classification = cf[1](X[i])
			if cf_classification != Y[i]:
				# output[cf].append(X[i])
				output[cf].append(key_from_value(ids_to_points,X[i]))

	return output

test_make_classifiers_to_misclassified = test_make_classifiers_to_misclassified(test_X,test_Y,\
	test_classifiers,test_ids_to_points)


def approx_equal(x,y,epsilon=0.001):
	"""
	Helper function that returns True if x and y are approximately equal
	"""
	return abs(x-y) <= epsilon

def choose_best_classifier(classifiers_to_misclassified,points_to_weight):
	"""
	Returns a tuple containing the name of the best classifier and its error on the current data set and weights
	"""
	best_classifier = -1
	best_error = 0.5
	for cf in classifiers_to_misclassified.keys():
		current_error = 0.0
		for misclassified_point in classifiers_to_misclassified[cf]:
			current_error += points_to_weight[misclassified_point]
		if abs(0.5-current_error) > abs(0.5-best_error):
			best_classifier = cf
			best_error = current_error

	if approx_equal(best_error,0.5):
		# No good classifier
		return (None,None)

	return (best_classifier,best_error)

def get_incorrect_classifications_overall(H,classifiers_to_misclassified,points):
	"""
	Returns a list of misclassified points 
	"""
	misclassified_points = []
	for point in points:
		wrong = 0.0
		right = 0.0
		for cf,alpha in H:
			if point in classifiers_to_misclassified[cf]:
				wrong += alpha
			else:
				right += alpha

		if wrong >= right:
			misclassified_points.append(point)

	return set(misclassified_points)



def adaboost(points,classifiers_to_misclassified,num_mistakes_tolerated=0,max_num_rounds=float("inf")):
	"""
	Executes the adaboost algorithm.

	Arguments:
		X:	training data
		Y:	training labels
		classifiers_to_misclassified:	dictionary mapping possible classifiers to the points that they misclassify
		use_smallest_error:		if True, pick the classifier with the smallest error, else if False pick one farthest from 1/2
		num_mistakes_tolerated:		Number of mistakes total classifier can make on data before termination
		max_num_rounds:		maximum number of rounds before termination

	Returns:
		H:	returned classifier that is (classifier,alpha) tuples
	"""

	# Parameter values and helper variables for algorithm
	N = len(points)
	classifiers_to_error_rate = {} # Dictionary of classifiers to the error rate of classifiers
	H = []

	# Initiliaze weights to be uniform
	points_to_weight = dict(zip(points,[1.0/N for i in range(N)]))

	if max_num_rounds == float("inf"):
		max_num_rounds = 100 # hardcoded value

	## New addition
	original_classifiers_to_misclassified = classifiers_to_misclassified.copy()

	# Main clause of algorithm
	for i in range(max_num_rounds):
		# Calculate error rates
		# for i in classifiers_to_misclassified.keys():
		# 	classifiers_to_error_rate[i] = sum([points_to_weight[j] for j in classifiers_to_misclassified[i]])

		# Choose classifier with best error rate
		(classifier,error_rate) = choose_best_classifier(classifiers_to_misclassified,points_to_weight)
		print (classifier,error_rate)
		if not classifier:
			return H

		# Compute alpha
		if approx_equal(error_rate,1.0):
			return -1.0*float("inf")
		elif approx_equal(error_rate,0.0):
			return float("inf")

		alpha = 0.5*math.log((1.0-error_rate)/error_rate)

		# Append to classifier
		H.append((classifier,alpha))

		# Compute weights for next step
		for point in points_to_weight.keys():
			if point in classifiers_to_misclassified[classifier]:
				# Misclassified point
				points_to_weight[point] = 0.5*points_to_weight[point]/error_rate
			else:
				# Correctly classified point
				points_to_weight[point] = 0.5*points_to_weight[point]/(1.0-error_rate)

		# Check terminating conditions
		# if len(get_incorrect_classifications_overall(H,classifiers_to_misclassified,points)) <= num_mistakes_tolerated:
		if len(get_incorrect_classifications_overall(H,original_classifiers_to_misclassified,points)) <= num_mistakes_tolerated:
			return H

		# NEW ADDITION
		classifiers_to_misclassified.pop(classifier)

	return H

test_points = test_ids_to_points.keys()
# print 'test_points',test_points
# print 'cf to miss',test_make_classifiers_to_misclassified
# print adaboost(test_points,test_make_classifiers_to_misclassified)

