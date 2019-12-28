import os
import sys
from os.path import expanduser
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
import tempfile
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y


home = expanduser("~")
os.environ.setdefault("LASER",home+"/projects/LASER/")
assert os.environ.get('LASER'), 'Please set the enviornment variable LASER'
LASER = os.environ['LASER']
print(LASER)
sys.path.append(LASER + '/source/lib')
sys.path.append(LASER+"/source/")

from text_processing import Token, BPEfastApply


import embed
from embed import *


class LASERClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, base_classifier = "knn",source_lang=None,target_lang=None,params={}):
        self.base_classifier = base_classifier
        self.source_lang  = source_lang
        self.target_lang  = target_lang
        self.params = params
        
        if base_classifier =="knn":
            self.clf = KNeighborsClassifier(**params)
        #elif base_classifier=="mlp" #TODO: add support for an MPL classifier
        else:
            raise ValueError("Unknown base classifier")

    # Function for encoding senteces using the LASER model. Code was adapted from 
    # Arguments:
    # in_file: path to file with the sentences
    # lang: the language to encode
    def _vectorize(self,in_file,lang):

        embedding = ''
        if lang is None or not lang:
            lang = "en"
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

            if lang != '--':
                tok_fname = tmpdir / "tok"
                Token(str(in_file),
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

    def fit(self, X, y):

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        tfile = tempfile.NamedTemporaryFile()
        np.savetxt(tfile.name,X,fmt="%s")
        
        X_laser = self._vectorize(tfile.name,self.source_lang)
        self.n_samples, self.n_features = X_laser.shape
        
        # Check that X and y have correct shape
        X_laser, y = check_X_y(X_laser, y,accept_sparse=False)
        
        self.clf.fit(X_laser,y)
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        #X = check_array(X)
        tfile = tempfile.NamedTemporaryFile()
        np.savetxt(tfile.name,X,fmt="%s")
        
        X_laser = self._vectorize(tfile.name,self.target_lang)
        predictions = self.clf.predict(X_laser)
        
        return predictions



