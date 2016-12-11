from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.svm import SVC

# Iris example
# from here: http://scikit-learn.org/stable/modules/feature_selection.html
digits = load_digits(n_class=2)
X,y = digits.data,digits.target

result = SelectKBest(chi2, k=10).fit(X,y)
X_new = result.fit_transform(X,y)


# Can change to just X to see how compares to baseline -> score is 0.878 with SVM, rbf kernel, k = 10
# is also 0.978 with linear kernel
X_train,y_train = X_new[:180,],y[:180]# Train data
X_test,y_test = X_new[180:,],y[180:,] # Test data

model = SVC(kernel='linear')
fitted_SVM = model.fit(X_train,y_train)
# print fitted_SVM.score(X_test,y_test)

############
# print 'scores',result.scores_
# # print 'pvalues',result.pvalues_

# scores_list = result.scores_.tolist()
# print [i[0] for i in sorted(enumerate(scores_list), key=lambda x:x[1])]

# scores_no_nans = result.scores_[~np.isnan(result.scores_)]
# print np.argsort(result.scores_)
# print scores_list[56] # 56 is the first nan in this list
################

###################### Compare to features selected by boosting
# [5, 47, 60, 7, 27, 8, 22, 48, 50, 32, 13, 56, 54, 0, 42, 39, 11, 31, 34, 23] original, keeping the nans ones
best_features = [5, 60, 27, 22, 50, 13, 54, 42, 11, 34]
best_features = best_features[:5]
features_to_remove = list(set(range(64))-set(best_features))

X = np.delete(X,[features_to_remove],axis=1)
X_train,y_train = X[:180,],y[:180]# Train data
X_test,y_test = X[180:,],y[180:,] # Test data

model = SVC(kernel='linear')
fitted_SVM = model.fit(X_train,y_train)
print fitted_SVM.score(X_test,y_test)

# Accuracy is 0.95 with k = 10 and linear kernel
# Accuracy is 0.78 with k=10 and rbf
# Accuracy is 0.92 with k = 5 and linear kernel
# Accuracy is 0.9 with k = 5 and rbf


# TODO also do this for random forest or decision trees to see if non-linearity helps (SVMs basically only linear)
	# if you find something that non-linearity helps, can say that that's better for the convolutional neural nets and stuff
	