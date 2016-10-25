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
	####
    # transicion inicial
    if i==1:
        feature_dict['trans_ini'] = history[i-1]
    elif i>2:
        # transicion
        feat_name = history[i-2]+'::'+history[i-1]
        feature_dict['trans'] = feat_name
        
        # trancision final
        if i==len(sentence)-1:
            feature_dict['trans_fin'] = history[i-1]

    #features de emision
	 ####
    if i>0:
        feat_name =    word +'::'+ history[i-1]
    else:
        feat_name = word
    feature_dict['emision'] = feat_name

    #  emision de sufijos
    for degree in range(1,4):
        suffix = word[-degree:]
        if i>0:
            feat_name = suffix + '::'+history[i-1]
        else:
            feat_name = suffix
        feature_dict['emis_suff_'+str(degree)] = feat_name
    
    # emision de ortografia
    if re.search('[A-Z].*',word):
        feature_dict['ort'] = 'prim_mayus'
    if re.search('[A-Z]+',word):
        feature_dict['ort'] = 'todo_mayus'

    return feature_dict















