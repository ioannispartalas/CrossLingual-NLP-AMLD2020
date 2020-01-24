import pandas as pd
import numpy as np
from .utils import load_embeddings,fit_vocab,sort_embeddings
from ast import literal_eval
from collections import Counter
from sklearn.metrics import accuracy_score,f1_score

class Dataset:
    """Experiment class, that reads data in raw format and prints stats.
    
    Arguments:
    ----------
    path: the path to the datasets
    source_lang: the source language, e.g. "en" for english
    target_lang: the target language, e.g. "nl" for dutch
    """
    def __init__(self, path,source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.tr_path = path +"semeval15.%s.train.csv" % source_lang
        self.te_path = path +"semeval15.%s.test.csv" % target_lang
    
    @staticmethod
    def read_csv(path):
        df = pd.read_csv(path)
        df['polarities'] = df['polarities'].apply(lambda l: literal_eval(l))
        df = df.loc[df.polarities.astype(bool)]
        df['sentiment'] = df['polarities'].apply(lambda l: Counter(l).most_common(1)[0][0])
        # Remove neutral class
        df = df[df.sentiment.isin(["positive","negative"])]
        return df[['text', 'sentiment']]

    def load_data(self):
        training = self.read_csv(self.tr_path)
        test = self.read_csv(self.te_path)
        print("\nTraining data\n==========")
        self.calculate_stats(training)
        print("\nTraining data\n==========")
        self.calculate_stats(test)
        self.train, self.y_train = training.text.values, training.sentiment.values
        self.test, self.y_test = test.text.values, test.sentiment.values

    
    def load_cl_embeddings(self,path_to_embeddings,dimension,skip_header):
        """
        Function to load the Cross-lingual embeddings for each language.
        
        Arguments:
        ----------
        path_to_embeddings: path to the cross-lingual embeddings
        dimension: an integer to set the dimension of the embeddings
        skip_header: boolean, whether to skip the first line or not
        """
        self.vocab_source = fit_vocab(self.train)
        self.vocab_target = fit_vocab(self.test)
        
        # full vocabulary
        self.vocab_ = fit_vocab(np.concatenate((self.train,self.test)))
        
        self.source_embeddings = load_embeddings(path_to_embeddings+"concept_net_1706.300."+self.source_lang, dimension,skip_header=skip_header,vocab=self.vocab_)
        self.target_embeddings = load_embeddings(path_to_embeddings+"concept_net_1706.300."+self.target_lang, dimension,skip_header=skip_header,vocab=self.vocab_)
        
        self.source_embeddings = sort_embeddings(self.source_embeddings,self.vocab_)
        self.target_embeddings = sort_embeddings(self.target_embeddings,self.vocab_)
        
        
    def calculate_stats(self, df):
        print("Training Data Shape: ", df.shape)
        print("Class distribution: ", df.sentiment.value_counts().to_dict())

        
class Runner:
    def __init__(self, pipeline, experiment):
        self.pipeline = pipeline
        self.experiment = experiment
        #self.experiment.load_data()
        
    def score(self, preds):
        #return accuracy_score(exp.y_test, preds)
        return f1_score(self.experiment.y_test, preds,average="binary",pos_label="positive")
    
    def eval_system(self,prefit=False,**kwargs):
        if prefit==False:
            self.pipeline.fit(self.experiment.train, self.experiment.y_train,**kwargs)
        
        preds = self.pipeline.predict(self.experiment.test,**kwargs)
        scores = self.score(preds)
        return scores
