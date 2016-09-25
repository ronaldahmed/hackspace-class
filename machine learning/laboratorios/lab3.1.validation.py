"""
Lab 3.1. Validación de Modelos de Clasificación
"""
 
import warnings
import ipdb
  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm, grid_search
from sklearn. linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cross_validation import train_test_split
from sklearn import datasets

warnings.filterwarnings("ignore")
##################################################################################################

 
def run_classifier(model,Xtrain,Ytrain, Xtest, Ytest, parameters={}, classes=[]):
    score = 'f1'
    gridCV = grid_search.GridSearchCV(model, parameters, scoring=score, cv = 5)
    gridCV.fit(Xtrain,Ytrain)

    print("[Val set] Best %s: %.4f" % (score,gridCV.best_score_))
    print("[Val set] Best parameters:")
    print(gridCV.best_params_)
    
    pred_test = gridCV.predict(Xtest)

    print("Metrics Testing data...")
    print(classification_report(Ytest, pred_test, target_names=classes))
    print("[Test set] Accuracy: %.4f" % accuracy_score(Ytest,pred_test))

    return gridCV.best_params_
 


##################################################################################################
 
if __name__ == '__main__':
    #################################################################################
    # Cargar el dataset:
    digits = datasets.load_digits()
    X, Y = digits.data, digits.target

    #################################################################################
    # Porcentaje de datos para testeo
    test_perc = 0.2 

    # Division de dataset en training y testing
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_perc,random_state=42)

    print("Muestras en training set: %i" % X_train.shape[0])
    print("Muestras en testing set:  %i" % X_test.shape[0])
 
    classes = [str(dig) for dig in range(10)]
    ##################################################################################################
    # LOGISTIC REGRESSION
     
    print("##################   Logistic Regression ################")
    logreg = LogisticRegression()

    parameters = {'C':[.005,0.008,0.01,0.015,0.02]}
     
    best_params = run_classifier(logreg, X_train, Y_train, X_test, Y_test, parameters, classes)
    

    ##################################################################################################
    # SUPPORT VECTOR MACHINE
    print("##################   SVM  ################")
    svc = svm.SVC()

    parameters = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear','rbf'],
        'gamma' : 10.0**-np.arange(1,4)
    }
    best_params = run_classifier(svc, X_train, Y_train, X_test, Y_test, parameters,classes)

    ##################################################################################################
    # RANDOM FORESTS
    print("##################   RF")
    rf = RF()
    parameters = dict(
        n_estimators=[10,50,80,100,200],
        criterion=['gini','entropy'],
        max_depth=[None,10,20],
    )
    best_params = run_classifier(rf, X_train, Y_train, X_test, Y_test, parameters, classes)
















