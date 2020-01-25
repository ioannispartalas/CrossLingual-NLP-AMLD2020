from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, confusion_matrix
import codecs
import numpy as np
import os
import pandas as pd
from collections import Counter

def load_embeddings(path, dimension,skip_header=True,vocab=None):
    """Simple function to load embeddings from txt file where one line one word embeddings in the form 
    word_token_1 .. word_token_n 300 numbers e.g.,
    
    new york 0.01 0.03 ... 0.04
    paris 0.011 0.02 ... 0.041
    
    Args 
    -----
    path: path to embeddings
    dimension: dimension of embeddings
    skip_header: whether to skip or not the header
    vocab: load only words in the vocabulary to save memory space
    
    Returns
    -------
    dictionary of word: embedding 
    """
    
    with codecs.open(path,"r",encoding="utf-8") as f:
        if skip_header==True:
            info = f.readline()
            print("{} vectors of dimension {}".format(*info.split()))
        vectors = {}
        for cnt,line in enumerate(f):
            elems = line.split()
            word = " ".join(elems[:-dimension])
            if vocab is not None and word in vocab:
                vectors[" ".join(elems[:-dimension])] =  np.array(elems[-dimension:]).astype(np.float)
            if vocab is None:
                vectors[" ".join(elems[:-dimension])] =  np.array(elems[-dimension:]).astype(np.float)
        print("Loaded {} vectors".format(len(vectors)))
        return vectors
    

# Fit the vocabulary of source and target documents
def fit_vocab(corpora):
    # Total vocabulary of the train/test corpora
    all_words = CountVectorizer(lowercase=True).fit(corpora).vocabulary_ 
    return all_words

def emb2numpy(embeddings_dico):
    keys = []
    values = []
    for k in embeddings_dico:
        keys.append(k)
        values.append(embeddings_dico[k])
    return keys,np.array(values)


def sort_embeddings(X_emb,word2index):
    """Function to sort the array of embeddings according to the index
    of the vocabulary.
    
    
    Arguments:
    ------------
    X_emb: the embeddings dictionary which holds a vector for each word
    word2index: mapping of word to index after fitting the vocabulary of the documents
    
    Returns:
    ------------
    A numpy array of same shape with the sorted embeddings.
    """
    sorted_word2index = sorted(word2index.items(), key=lambda kv: kv[1])
    sortedEmbed  = []
    for k in sorted_word2index:
        try:
            sortedEmbed.append(X_emb[k[0]])
        except:
            sortedEmbed.append(np.zeros(300))
    return np.array(sortedEmbed)

def format_score(x):
    return "{:.2f}".format(x*100.0)

def load_language(language = 'en', train_or_test = 'train'):
    """
    load dataset for a particular language and dataset (train or test)
    """
    path = os.environ['WORKDIR'] + 'data/laser/'
    feat_fn =  path  + language + '_laser_' + train_or_test + '.npy'
    label_fn = path  + language + '_' + train_or_test + '_labels_adan.txt'
    labels = pd.read_csv(label_fn,header=None).values.ravel() 
    #kk = np.squeeze(np.where(labels != 2))
    feat = np.load(feat_fn)
    #labels = labels[kk]
    return feat,labels

def load_training_languages(languages):
  """ 
  return concatenated features and labels for all languages specified in the list languages
  """
  x_train,y_train= [],[]

  for lang in languages:
    x_tr,y_tr = load_language(lang,'train')  
    x_train.append(x_tr)
    y_train.append(y_tr)

  x_train =np.vstack(x_train)
  y_train =np.hstack(y_train)
  return x_train,y_train


def model_evaluation(model, languages):
  """
  Measure F1 score and confusion matrix for the provided model  over test data for the specified languages
  model must have a predict method returning the classes of the tested data
  """
  
  EVAL = {}
  for lang in languages:
    x_test,y_test = load_language( lang, 'test')
    y_pred = model.predict(x_test)

    F1 = f1_score (y_test,y_pred, pos_label = 'negative')
    CONF =  pd.DataFrame(confusion_matrix(y_test,y_pred),index = ['TRUE NEGATIVE','TRUE POSITIVE'],columns=('PRED NEGATIVE','PRED POSITIVE'))

    EVAL[lang] = (F1,CONF)
  for lang, metric in EVAL.items():
    print(lang,': F1= ', metric[0],'\n', metric[1],'\n')
  return EVAL


def get_statistics(languages):
   for lang in languages:
        
         print(lang,': ')
         _,y = load_language( lang, 'train')
         c = Counter(y)
         print('Training:'  , c.most_common(10), ', total =',sum(c.values())  )
         _,y = load_language( lang, 'test')
         c = Counter(y)
         print('Test:'  , c.most_common(10), ', total =',sum(c.values()), '\n'  )
  
