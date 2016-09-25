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
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
        feature_dict['trans_init'] = "<START>"
    else:
        prevword, prevpos = sentence[i-1]
        y_prev = history[i-1]
        feature_dict['trans'] = y_prev
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
        feature_dict['trans_final'] = "<END>"
    
    #features de emision
    feature_dict['emision'] = word
    feature_dict['emision_pos'] = pos

       # sufijos y prefijos
    for b in range(1,4):
      suf = word[-b:].lower()
      pref = word[:b]
      feature_dict['suf'+str(b)] = suf
      feature_dict['pref'+str(b)] = pref

    #Ortografia
    initCap = re.compile('[A-Z][a-z]+\.*')
    match = initCap.match(word)
    if match:
      feature_dict['initCap'] = 'initCap'
   
    
    return feature_dict





