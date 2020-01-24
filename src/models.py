import os
import sys
from os.path import expanduser
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from sklearn.neighbors import KNeighborsClassifier
import tempfile
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


home = expanduser("~")
os.environ.setdefault("LASER",home+"/projects/LASER/")
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']

sys.path.append(LASER + '/source/lib')
sys.path.append(LASER+"/source/")

from text_processing import Token, BPEfastApply


import embed
from embed import *


class LASERClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier = KNeighborsClassifier(n_neighbors=2) ,source_lang=None,target_lang=None):
        self.base_classifier = base_classifier
        self.source_lang  = source_lang
        self.target_lang  = target_lang
        self.clf = base_classifier


    def fit(self, X, y):

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        self.n_samples, self.n_features = X.shape
        
        # Check that X and y have correct shape
        X, y = check_X_y(X, y,accept_sparse=False)
        
        self.clf.fit(X,y)
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)
        predictions = self.clf.predict(X)
        
        return predictions

    
    
class Doc2Laser(BaseEstimator, TransformerMixin):
    """Transform raw documents to their LASER representations.
    
    Parameters:
    -------------
    lang: the language to encode
    """
    def __init__(self,lang=None):
        """
        lang: the language to encode, for example "en" (english)
        """
        self.lang = lang

    def fit(self, X, y=None,**fit_params):
        return self
    
    def _vectorize(self,docs):
        """
        Function for encoding senteces using the LASER model. Code was adapted from 
        
        Arguments:
        docs: the documents to encode, an iterable
        lang: the language to encode
        """
        embedding = ''
        if self.lang is None or not self.lang:
            lang = "en"
            print("Warning: using default language English")
        else:
            lang = self.lang
            
        # encoder
        
        
        model_dir = os.environ.get('LASER')+"models"
        encoder_path = model_dir +"/" + "bilstm.93langs.2018-12-26.pt"
        bpe_codes_path = model_dir+"/"+  "93langs.fcodes"
        print(f' - Encoder: loading {encoder_path}')
        encoder = SentenceEncoder(encoder_path,
                                  max_sentences=None,
                                  max_tokens=12000,
                                  sort_kind='mergesort',
                                  cpu=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)

            bpe_fname = tmpdir / 'bpe'
            bpe_oname = tmpdir / 'out.raw'

            temp_infile = tmpdir / 'temp_in_docs.txt'
            np.savetxt(temp_infile,docs,fmt="%s")
            
            if lang != '--':
                tok_fname = tmpdir / "tok"
                Token(str(temp_infile),
                      str(tok_fname),
                      lang=lang,
                      romanize=True if lang == 'el' else False,
                      lower_case=True,
                      gzip=False,
                      verbose=True,
                      over_write=False)
                ifname = tok_fname

            BPEfastApply(str(ifname),
                         str(bpe_fname),
                         str(bpe_codes_path),
                         verbose=True, over_write=False)
            ifname = bpe_fname
            EncodeFile(encoder,
                       str(ifname),
                       str(bpe_oname),
                       verbose=True,
                       over_write=False,
                       buffer_size=10000)
            dim = 1024
            X = np.fromfile(str(bpe_oname), dtype=np.float32, count=-1)
            X.resize(X.shape[0] // dim, dim)
            embedding = X

            return X
    
    def transform(self, X):
        """
        This function transforms the raw representations in the iterable X using LASER.
        
        Arguments:
        ----------
        X: an iterable of the raw documents.
        
        Returns:
        ----------
        A numpy array of shape (X.shape[0],1024)
        """
        X_laser = self._vectorize(X)
        return X_laser

    
class nBowClassifier(BaseEstimator, ClassifierMixin):
    """Model that averages the cross-lingual representations in the document and learns a classifier on top of it.
    
    Arguments:
    ----------
    base_classifier: the classifier that will be used to train on the source language. It accepts any classifier that implements fit and predict API of scikit-learn.
    V_source: a numpy array that contains the vector representation of the source documents.
    V_target: a numpy array that contains the vector representation of the target documents.
    params: optional parameters for the classifier.
    """

    def __init__(self, base_classifier = KNeighborsClassifier(n_neighbors=2),V_source=None,V_target=None,params={}):
        self.base_classifier = base_classifier
        self.V_source = V_source
        self.V_target = V_target
        self.params = params
        self.clf = base_classifier

    # neural bag-of-words baseline
    # average word embeddings of each document
    # V_emb: this holds the embedding of each word
    # X: the vectorized array of documents. Note that the indices of the features should correspond to the same indices in the V_emb array
    def _nBOW(self,V_emb,X):
        X_avg = []
        for doc in X:
            doc_vecs = V_emb[doc.indices,:]
            avg_vec = np.sum((doc_vecs*doc.data[:,np.newaxis]),axis=0)/(doc.data.sum() + 1.0)
            X_avg.append(avg_vec)
        
        return np.array(X_avg)
    
    def fit(self, X, y):

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        X_avg = self._nBOW(self.V_source,X)
        self.n_samples, self.n_features = X_avg.shape
        
        # Check that X and y have correct shape
        X_avg, y = check_X_y(X_avg, y,accept_sparse=False)
        
        self.clf.fit(X_avg,y)
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)
        X_avg = self._nBOW(self.V_target,X)
        predictions = self.clf.predict(X_avg)
        
        return predictions