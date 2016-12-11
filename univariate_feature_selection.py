from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.svm import SVC

# Iris example
# from here: http://scikit-learn.org/stable/modules/feature_selection.html
digits = load_digits(n_class=2)
X,y = digits.data,digits.target

print '#### Chi squared ###'
def chi_squared_features():
	for i in range(1,10):
		result = SelectKBest(chi2, k=i).fit(X,y)
		# result = SelectKBest(f_classif, k=i).fit(X,y)
		X_new = result.fit_transform(X,y)
		# Can change to just X to see how compares to baseline -> score is 0.878 with SVM, rbf kernel, k = 10
		# is also 0.978 with linear kernel
		X_train,y_train = X_new[:180,],y[:180]# Train data
		X_test,y_test = X_new[180:,],y[180:,] # Test data

		# model = SVC(kernel='linear') # SVM
		# model = RandomForestClassifier()
		model = DecisionTreeClassifier()
		fitted_SVM = model.fit(X_train,y_train)
		# print 'best',i,'features: ',fitted_SVM.score(X_test,y_test)
		print '\hline'
		print '$k$ = ',i, '& ',fitted_SVM.score(X_test,y_test), '&'

# chi_squared_features()
# print fitted_SVM.score(X_test,y_test)

############
# result = SelectKBest(chi2, k=10).fit(X,y)

# print 'scores',result.scores_
# # print 'pvalues',result.pvalues_

# scores_list = result.scores_.tolist()
# print [i[0] for i in sorted(enumerate(scores_list), key=lambda x:x[1])]

# scores_no_nans = result.scores_[~np.isnan(result.scores_)]
# print np.argsort(result.scores_)
# print scores_list[56] # 56 is the first nan in this list
# resulting features are [36,28,44,35,27,20,43,38,30,46,19]
################

###################### Compare to features selected by boosting
# [5, 47, 60, 7, 27, 8, 22, 48, 50, 32, 13, 56, 54, 0, 42, 39, 11, 31, 34, 23] original, keeping the nans ones
# new features: [52, 24, 43, 51, 20, 63, 55, 6, 57, 61]

print '#### BOOSTING #####'
def test_boosting_features():
	# best_features = [5, 60, 27, 22, 50, 13, 54, 42, 11, 34]
	best_features = [52, 24, 43, 51, 20, 63, 55, 6, 57, 61]
	for i in range(1,10):
		new_features = best_features[:i]
		features_to_remove = list(set(range(64))-set(new_features))

		X_extra = X
		X_extra = np.delete(X_extra,[features_to_remove],axis=1)
		X_train,y_train = X_extra[:180,],y[:180]# Train data
		X_test,y_test = X_extra[180:,],y[180:,] # Test data

		# model = SVC(kernel='linear') # SVM
		# model = RandomForestClassifier()
		model = DecisionTreeClassifier()
		fitted_SVM = model.fit(X_train,y_train)
		# print 'best',i,'features: ',fitted_SVM.score(X_test,y_test)
		print '\hline'
		print fitted_SVM.score(X_test,y_test), '&'

# test_boosting_features()

def plot_decision_boundary():
	# for chi squared features
	result = SelectKBest(chi2, k=8).fit(X,y)
	X_new = result.fit_transform(X,y)
	# Can change to just X to see how compares to baseline -> score is 0.878 with SVM, rbf kernel, k = 10
	# is also 0.978 with linear kernel
	X_train,y_train = X_new[:180,],y[:180]# Train data
	X_test,y_test = X_new[180:,],y[180:,] # Test data

	model = SVC(kernel='linear') # SVM
	# model = RandomForestClassifier()
	# model = DecisionTreeClassifier()
	fitted_SVM = model.fit(X_train,y_train)
	predicted_class = fitted_SVM.predict(X_test)
	for i in range(len(X_test)):
		


