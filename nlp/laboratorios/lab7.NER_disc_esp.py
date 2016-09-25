import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import *
from nltk import UnigramTagger, BigramTagger, TrigramTagger
from sklearn.cross_validation import train_test_split
import readers.ancora as ancora
from modelos.seq_classifier import ConsecutiveNERTagger
import ipdb

##########################################################################
stemmer = SnowballStemmer('spanish')


##########################################################################

if __name__== "__main__":
    print("Leyendo data...")
    reader = ancora.AncoraCorpusReader()
    data = reader.pos_ne_tagged_sents()[:100]

    test_perc = 0.2
    train_set,test_set = train_test_split(data,test_size=test_perc,random_state=42)

    model = ConsecutiveNERTagger(train_set)

    print("Test ACC:%.4f" % model.evaluate(test_set)*100)
    
