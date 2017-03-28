import math
import numpy as np
import pandas
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import report

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))

#---------------Ready algorithms-----------------

names = [ "Output",
          "Alcohol",
          "Malic acid",
          "Ash",
          "Alcalinity of ash",
          "Magnesium",
          "Total phenols",
          "Flavanoids",
          "Nonflavanoid phenols",
          "Proanthocyanins",
          "Color intensity",
          "Hue",
          "OD280/OD315 of diluted wines",
          "Proline"
        ]
dataset = pandas.read_csv("../data/wine.data", names=names)
print("Ilosc przykladow: " + str(dataset.shape))
print(dataset.head(5))
#features descriptions
print(dataset.describe())
#class distribution
print(dataset.groupby('Output').size())
# box and whisker plots
output = dataset[["Output"]]
#print(output)
dataset_whole = dataset
dataset = dataset.drop(['Output'], axis=1)
df_norm = (dataset - dataset.mean()) / (dataset.max() - dataset.min())
df_norm.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
df_norm.plot(kind='density', subplots=True, layout=(2,7), sharex=False)
plt.show()

#dataset.plot(kind='box', subplots=True, layout=(2,7), sharex=False, sharey=False)
plt.show()
# histograms
#dataset.hist()
df_norm.hist()
plt.show()
# scatter plot matrix
#scatter_matrix(df_norm)
#plt.show()

df_norm['Output'] = output
#print("After adding Output again:")
#print(df_norm.head(5))
# Split-out validation dataset
array = df_norm.values
X = array[:,0:12]
Y = array[:,13]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

#print(X_train)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
report_str = classification_report(Y_validation, predictions)
report.plot_classification_report(report_str)

plt.savefig('lda.png', dpi=200, format='png', bbox_inches='tight')
plt.close()
#----------Neural network-----------------

print("Neural Network!")

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 6), random_state=1)

clf.fit(X_train, Y_train)

predictions = clf.predict(X_validation)

print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
report_str = classification_report(Y_validation, predictions)
report.plot_classification_report(report_str)

plt.savefig('mlp.png', dpi=200, format='png', bbox_inches='tight')
plt.close()
#--------Logistic Regression from scratch------

def Sigmoid(z):
	G_of_Z = float(1.0 / float((1.0 + math.exp(-1.0*z))))
	return G_of_Z 

def Hypothesis(theta, x):
	z = 0
	for i in xrange(len(theta)):
		z += x[i]*theta[i]
	return Sigmoid(z)

def Cost_Function(X,Y,theta,m):
	sumOfErrors = 0
	for i in xrange(m):
		xi = X[i]
		hi = Hypothesis(theta,xi)
		if Y[i] == 1:
			error = Y[i] * math.log(hi)
		elif Y[i] == 0:
			error = (1-Y[i]) * math.log(1-hi)
		sumOfErrors += error
	const = -1/m
	J = const * sumOfErrors
	#print 'cost is ', J 
	return J

def Cost_Function_Derivative(X,Y,theta,j,m,alpha):
	sumErrors = 0
	for i in xrange(m):
		xi = X[i]
		xij = xi[j]
		hi = Hypothesis(theta,X[i])
		error = (hi - Y[i])*xij
		sumErrors += error
	m = len(Y)
	constant = float(alpha)/float(m)
	J = constant * sumErrors
	return J

def Gradient_Descent(X,Y,theta,m,alpha):
	new_theta = []
	constant = alpha/m
	for j in xrange(len(theta)):
		CFDerivative = Cost_Function_Derivative(X,Y,theta,j,m,alpha)
		new_theta_value = theta[j] - CFDerivative
		new_theta.append(new_theta_value)
	return new_theta

def Logistic_Regression(X,Y,alpha,theta,num_iters):
	m = len(Y)
	for x in xrange(num_iters):
		new_theta = Gradient_Descent(X,Y,theta,m,alpha)
		theta = new_theta
		if x % 100 == 0:
			Cost_Function(X,Y,theta,m)
			#print 'theta ', theta	
			#print 'cost is ', Cost_Function(X,Y,theta,m)
        return theta

def MultiClassLogisticRegression(X,Y,alpha,iterations):
    initial_theta = [0,0]
    Y_1 = [1 if x == 1 else 0 for x in Y]
    Y_2 = [1 if x == 2 else 0 for x in Y]
    Y_3 = [1 if x == 3 else 0 for x in Y]

    Y_v1 = [1 if x == 1 else 0 for x in Y_validation]
    Y_v2 = [1 if x == 2 else 0 for x in Y_validation]
    Y_v3 = [1 if x == 3 else 0 for x in Y_validation]

    wine = []
    wine.append(Logistic_Regression(X,Y_1,alpha,initial_theta,iterations))
    wine.append(Logistic_Regression(X,Y_2,alpha,initial_theta,iterations))
    wine.append(Logistic_Regression(X,Y_3,alpha,initial_theta,iterations)) 

    predictions = []

    score1 = 0
    score2 = 0
    score3 = 0
    winner = ""
    length = len(X_validation)
    for i in xrange(length):
        flag = 0
     	prediction = round(Hypothesis(wine[0],X_validation[i]))
        #print("Pred 1: " + str(prediction))
    	answer = Y_v1[i]
    	if prediction == answer:
            if prediction == 1.0:
                predictions.append(1)
                flag = 1
    	    score1 += 1
     	prediction = round(Hypothesis(wine[1],X_validation[i]))
        #print("Pred 2: " + str(prediction))

    	answer = Y_v2[i]
    	if prediction == answer:
            if prediction == 1.0:
                predictions.append(2)
                flag = 1

    	    score2 += 1
     	prediction = round(Hypothesis(wine[2],X_validation[i]))
        #print("Pred 3: " + str(prediction))
    	answer = Y_v3[i]
    	if prediction == answer:
            if prediction == 1.0:
                predictions.append(3)
                flag = 1

            score3 += 1
        if flag == 0:
            predictions.append(1)
    my_score = float(score3 + score2 + score1) / float(3*length)

    return my_score,predictions

# setting variables
initial_theta = [0,0]
alpha = 0.1
iterations = 1000
score,predictions = MultiClassLogisticRegression(X_train,Y_train,alpha,iterations)
print("Your score: " + str(score))
clf = LogisticRegression()
clf.fit(X_train,Y_train)
scikit_score = clf.score(X_validation,Y_validation)
print("Scikit score: " + str(scikit_score))
print("Predictions len: " + str(len(predictions)) + " X validation len: " + str(len(X_validation)))
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
report_str = classification_report(Y_validation, predictions)
report.plot_classification_report(report_str)

plt.savefig('test_plot_classif_report.png', dpi=200, format='png', bbox_inches='tight')
plt.close()
