import modelos.multinomial_naive_bayes as mnb
from readers.sentiment_reader import SentimentCorpus
from sklearn.metrics import classification_report, accuracy_score
import ipdb


# Leyendo y procesando corpus
test_perc = 0.2
stemming = False
rare_thr = 5

sc = SentimentCorpus(test_perc=test_perc,
					 thr=rare_thr,
					 stemming=stemming)


# Inicializar modelo
smoothing = 1.0
model = mnb.MultinomialNaiveBayes(smooth=smoothing)
# Entrenamiento
params = model.train(sc.X_train,sc.Y_train)



# Evaluacion Train dataset
Ytrain_pred = model.test(sc.X_train,params)
print("Metrics training data...")
print(classification_report(sc.Y_train, Ytrain_pred, target_names=['pos','neg']))
print("[Training set] Accuracy: %.4f" % accuracy_score(sc.Y_train,Ytrain_pred))
print("#################################################################")

# Evaluacion Test dataset
Ytest_pred = model.test(sc.X_test,params)
print("Metrics test data...")
print(classification_report(sc.Y_test, Ytest_pred, target_names=['pos','neg']))
print("[Test set] Accuracy: %.4f" % accuracy_score(sc.Y_test,Ytest_pred))
