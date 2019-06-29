from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from joblib import dump, load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def classify(name,x_train,y_train,x_test = None, y_test = None):
    if name == "Support Vector Machine":
        name = "SVC"
    elif name == "Neural Network":
        name = "MLPClassifier"
    elif name == "Naive Bayes":
        name = "GaussianNB"
    else:
        name = name.replace(" ","")
    clf_str = name + "()"
    clf = eval(clf_str)
    clf.fit(x_train,y_train)
    if x_test is not None and y_test is not None:
        print(clf.score(x_test,y_test))
    dump(clf, 'clf.joblib')
    print("Classfier model stored in clf.joblib ")
    return clf

def regression(name,x_train,y_train, x_test = None, y_test = None):
    if name == "Support Vector Machine":
        name = "SVR"
    elif name == "Neural Network":
        name = "MLPClassifier"
    else:
        name = name.replace(" ","")
    reg_str = name + "()"
    reg = eval(reg_str)
    reg.fit(x_train,y_train)
    if x_test is not None and y_test is not None:
        print(reg.score(x_test,y_test))
    dump(reg, 'reg.joblib')
    print("Regression model stored as reg.joblib")
    return reg

def dimensionality_reduction(name,x_train,num_components):
    if name == "T-SNE":
        name = "TSNE"
    elif name == "Locally Linear Embedding":
        name = "LocallyLinearEmbedding"
    else:
        name = name.replace(" ","")


    dim_str = name + "(" + num_components + ")"
    dim = eval(dim_str)
    components = dim.fit_transform(x_train)
    print("Predictive model stored as reg.joblib")


if __name__ == '__main__':
    type = input("C or R:\n")
    if type == "C":
        test = classify(input("Enter Classifier: "),[[-1],[-2],[-3],[-0.001],[0.001],[1],[2],[3],[4],[5]],[0,0,0,0,1,1,1,1,1,1])
        num = input("Enter a number: \n")
        while(True):
            print(test.predict([[float(num)]]))
            num = input("Enter a number: \n")
    else:
        test = regression(input("Enter Regression Algorithm: "),[[0],[121.75], [365], [730]],[0,.5,1,2])
        num = input("Enter a number: \n")
        while(True):
            print(test.predict([[float(num)]]))
            num = input("Enter a number: \n")