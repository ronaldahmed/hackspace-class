import nltk
from nltk.metrics import accuracy
from nltk.tag.util import untag
import re
import ipdb

class ConsecutiveNERTagger(nltk.TaggerI): # [_consec-chunk-tagger]

    def __init__(self, train_sents):
        train_set = []

        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) # [_consec-use-fe]
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train( # [_consec-use-maxent]
            train_set, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

    def evaluate(self,gold):
        tagged_sents = [list(self.tag( untag(sent) )) for sent in gold]
        gold_tokens = sum(gold, [])
        test_tokens = sum(tagged_sents, [])
        return accuracy(gold_tokens, test_tokens)



def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    feature_dict = {}

    # features de transicion

    #features de emision

    # sufijos y prefijos

    #Ortografia

    return feature_dict















