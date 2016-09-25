import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.probability import *
from nltk import UnigramTagger, BigramTagger, TrigramTagger
from sklearn.cross_validation import train_test_split
import readers.ancora as ancora
import ipdb

##########################################################################
stemmer = SnowballStemmer('spanish')

# Estimadores
mle = lambda fd,bins: MLEProbDist(fd)
laplace = LaplaceProbDist
ele = ELEProbDist # Expected Likelihood Estimate
witten = WittenBellProbDist
gt = lambda fd, bins: SimpleGoodTuringProbDist(fd, bins=1e5)

##########################################################################
def lidstone(gamma):
    return lambda fd, bins: LidstoneProbDist(fd, gamma, bins)

##########################################################################

if __name__== "__main__":
    print("Leyendo data...")
    reader = ancora.AncoraCorpusReader()
    data = reader.ne_tagged_sents()

    ipdb.set_trace()

    data = data[:100]

    test_perc = 0.2
    train_set,test_set = train_test_split(data,test_size=test_perc,random_state=42)

    states_set  = list(set([tag for sent in data for (word,tag) in sent]))
    vocab = list(set([word for sent in train_set for (word,tag) in sent]))

    hmm_trainer = nltk.HiddenMarkovModelTrainer(states_set,vocab)

    ####################################################################################
    print("Entrenando modelos N-gram...")

    unigram_tagger = UnigramTagger(train_set)
    bigram_tagger = BigramTagger(train_set, backoff=unigram_tagger) # uses unigram tagger in case it can't tag a word
    trigram_tagger = TrigramTagger(train_set, backoff=bigram_tagger)

    ####################################################################################
    print("Entrenando HMM con estimadores escogidos...")
    print("---------------------------------------------")
    print("  Estimador: witten...")
    wit = hmm_trainer.train_supervised(train_set, estimator = witten)

    print("  Estimador: Good Turing...")
    gtt = hmm_trainer.train_supervised(train_set, estimator = gt)

    print("  Estimador: Lidstone...")
    lds = hmm_trainer.train_supervised(train_set, estimator = lidstone(0.8))
    
    ####################################################################################
    print("Evaluando modelos...")
    print("---------------------------------------------")
    print('UnigramTagger: %.4f %%' % (unigram_tagger.evaluate(test_set) * 100) )
    print('BigramTagger: %.4f %%' % (bigram_tagger.evaluate(test_set) * 100) )
    print('TrigramTagger: %.4f %%' % (trigram_tagger.evaluate(test_set) * 100) )

    print('HMM_witten: %.4f %%' % (wit.evaluate(test_set) * 100) )
    print('HMM_turing: %.4f %%' % (gtt.evaluate(test_set) * 100) )
    print('HMM_lidstone: %.4f %%' % (lds.evaluate(test_set) * 100) )

    

    
