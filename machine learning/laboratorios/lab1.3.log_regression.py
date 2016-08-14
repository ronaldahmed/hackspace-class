'''
[HackSpace] Introducción a Machine Learning
Laboratorio 1.2.
Regresión Logistica / Maximum Entropy Classifier
'''

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from modelos.max_ent_batch import MaxEnt_batch

#################################################################################
# Cargar el dataset:
digits = datasets.load_digits()
X, y = digits.data, digits.target
y = y.reshape(-1,1) # formato [Nx1]

#################################################################################
# Porcentaje de datos para testeo
test_perc = 0.2 

# Division de dataset en training y testing
indexes = range(X.shape[0])
train_idxs,test_idxs = train_test_split(indexes,test_size=test_perc,random_state=42)

Xtrain = X[train_idxs,:]
Ytrain = y[train_idxs,:]

Xtest = X[test_idxs,:]
Ytest = y[test_idxs,:]
print("Muestras en training set: %i" % Xtrain.shape[0])
print("Muestras en testing set:  %i" % Xtest.shape[0])


#################################################################################
# Configuracion de modelo
nr_epochs = 10
learning_rate = 1.0
averaged = True
regularizer = 1

# Inicializamos modelo
mext = MaxEnt_batch(regularizer=regularizer)
# Entrenamiento
mext.train(Xtrain,Ytrain)

#################################################################################
# Evaluacion
digit_names = [str(dig) for dig in range(10)]

# Evaluacion Train dataset
Ytrain_pred = mext.test(Xtrain,mext.w)
print("Metrics training data...")
print(classification_report(Ytrain, Ytrain_pred, target_names=digit_names))
print("[Training set] Accuracy: %.4f" % accuracy_score(Ytrain,Ytrain_pred))

print("#################################################################")

# Evaluacion Test dataset
Ytest_pred = mext.test(Xtest,mext.w)
print("Metrics test data...")
print(classification_report(Ytest, Ytest_pred, target_names=digit_names))
print("[Test set] Accuracy: %.4f" % accuracy_score(Ytest,Ytest_pred))

