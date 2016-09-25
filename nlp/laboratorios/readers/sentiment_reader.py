import os
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.cross_validation import train_test_split
import ipdb


_base_sentiment_dir = os.path.join(os.path.dirname(__file__),"..","datasets","sentiment")
RARE = '<R>'  # etiqueta para palabra de baja frecuencia

class SentimentCorpus:    
    def __init__(self,test_perc=0.2,thr=5,stemming=False):
        # Stemming setup
        self. stemmer = SnowballStemmer('english')
        self.stemming = stemming
        # umbral de palabras RARE
        self.rare_threshold = thr
        # Leyendo dataset
        corpus,y = self.read_corpus()
        train_corpus,test_corpus,y_train,y_test = train_test_split(corpus,y,test_size=test_perc,random_state=42)

        # extraer features de data de entrenamiento
        self.feat_counts = {}
        self.feat_dict = {}
        self.n_feats = 0
        self.build_features(train_corpus)

        # representas cada muestra como caracteristicas
        self.X_train = self.get_features(train_corpus)
        self.X_test  = self.get_features(test_corpus)
        self.Y_train = y_train
        self.Y_test  = y_test


    def read_corpus(self):
        vocab = {}
        corpus_y = []
        pos_file = open(os.path.join(_base_sentiment_dir,"positive.review"))
        for line in pos_file:
            line = line.strip('\n')
            if line=='':
                continue
            doc = {}
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name,counts = feat.split(":")
                doc[name] = int(counts)
            corpus_y.append( (doc,1) )
        pos_file.close()
        
        neg_file = open(os.path.join(_base_sentiment_dir,"negative.review"))
        for line in neg_file:
            line = line.strip('\n')
            doc = {}
            if line=='':
                continue
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name,counts = feat.split(":")
                doc[name] = int(counts)
            corpus_y.append( (doc,0) )
        neg_file.close()
        
        np.random.shuffle(corpus_y)
        corpus = [xy[0] for xy in corpus_y]
        y = np.array([xy[1] for xy in corpus_y])
        y = np.reshape(y,[-1,1])

        return corpus,y

    def build_features(self,corpus):
        vocab = {}
        for doc in corpus:
            for w,f in doc.items():
                if w not in vocab:
                    vocab[w]=0
                vocab[w]+=f

        self.feat_counts = {}
        self.feat_dict = {}
        self.n_feats = 0
        for w,f in vocab.items():
            if self.stemming:
                w = self.stemmer.stem(w)
            if f<=self.rare_threshold:
                w = RARE
            if w not in self.feat_dict:
                self.feat_dict[w] = self.n_feats
                self.feat_counts[w] = 0
                self.n_feats+=1
            self.feat_counts[w]+=f
        

    def get_features(self,corpus):
        len_corpus = len(corpus)
        X = np.zeros([len_corpus,self.n_feats])
        for i,doc in enumerate(corpus):
            for token,freq in doc.items():
                if self.stemming:
                    token = self.stemmer.stem(token)
                if freq <= self.rare_threshold or token not in self.feat_dict:
                    token = RARE
                feat_id = self.feat_dict[token]
                X[i,feat_id] = freq
        return X










