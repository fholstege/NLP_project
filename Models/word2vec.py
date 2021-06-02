import multiprocessing
from gensim.models import Word2Vec
from nltk import tokenize
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib as plt
import numpy as np
from nltk.stem import WordNetLemmatizer 
from pandas.core.common import flatten


# read merged files, get body of text, adapt index

# stack data from all journals
df = pd.read_parquet('../Data/clean/data_journals_adject_nouns_merged.gzip')
df = df.reset_index()
body_texts = df['body'].tolist()

# split the texts
texts = [row.split() for row in df['body_lemmatized']]




# create train and test set, save which ones are train and test
random_indeces = list(np.random.permutation(len(texts)))
train_size = int(round(len(texts) * 0.7,0))
train_indeces = random_indeces[:train_size]
test_indeces = random_indeces[train_size:]
train_bool =  [1 if i in train_indeces else 0 for i in range(0,max(random_indeces)+1)]
train_texts = [texts[i] for i in train_indeces]
test_texts = [texts[j] for j in test_indeces]


# get the number of cores
cores = multiprocessing.cpu_count()


# window: 5, why? what other words (of any type) are used in related discussions? Smaller windows tend to capture more about word itself: what other words are functionally similar?
# https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf
window = 5
# size = dimensionality of the vector; needs to be a lot lower than the vocabulary
# mikolov: 300
size = 300 

# alpha: learning rate
alpha = 0.025 

# based on mikolov et al; negative sampling
negative = 5


# train word2vec on training set
w2v_model = Word2Vec(train_texts,
                     window=window,
                     size=size,
                     alpha = alpha,
                     negative = negative,
                     workers=cores-1)




# summarize for couple of common words which words are similar
w2v_model.most_similar('retail')
w2v_model.most_similar('effect')


# get document representations
def document_vector_mean(doc, w2v_model):
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in w2v_model.wv.vocab]
    return np.mean(w2v_model[doc], axis=0)

def document_vector_max(doc, w2v_model):
    """Create document vectors by max of word vectors. Remove out-of-vocabulary words."""

    doc = [word for word in doc if word in w2v_model.wv.vocab]
    return np.max(w2v_model[doc], axis=0)

def document_vector_min(doc, w2v_model):
    """Create document vectors by min of word vectors. Remove out-of-vocabulary words."""

    doc = [word for word in doc if word in w2v_model.wv.vocab]
    return np.min(w2v_model[doc], axis=0)



list_doc_representation_mean = [document_vector_mean(doc, w2v_model) for doc in texts ]
list_doc_representation_max = [document_vector_max(doc, w2v_model) for doc in texts ]
list_doc_representation_min = [document_vector_min(doc, w2v_model) for doc in texts ]


names_mean = ['mean_dim_' + str(i) for i in range(1,size + 1)]
names_max = ['max_dim_' + str(i) for i in range(1,size + 1)]
names_min = ['min_dim_' + str(i) for i in range(1,size + 1)]


df_doc_representation_mean = pd.DataFrame(list_doc_representation_mean)
df_doc_representation_max = pd.DataFrame(list_doc_representation_max)
df_doc_representation_min = pd.DataFrame(list_doc_representation_min)

df_doc_representation_mean.columns = names_mean
df_doc_representation_max.columns = names_max
df_doc_representation_min.columns = names_min


df_doc_representation_mean['Cited by']= df['Cited by']
df_doc_representation_max['Cited by']= df['Cited by']
df_doc_representation_min['Cited by']= df['Cited by']


df_doc_representation_mean['train_set'] = train_bool
df_doc_representation_max['train_set'] = train_bool
df_doc_representation_min['train_set'] = train_bool


df_doc_representation_mean.to_parquet('../data/representations/word2vec_doc_representation_mean.gzip',compression='gzip')
df_doc_representation_max.to_parquet('../data/representations/word2vec_doc_representation_max.gzip', compression='gzip')
df_doc_representation_min.to_parquet('../data/representations/word2vec_doc_representation_min.gzip', compression='gzip')















# get the accuracy
## this takes very long to calculate

#with open("test_data_window.txt", 'w', encoding="utf-8") as f:
    
#    f.write(': articles \n')
    
#    for text_list in test_texts:
#        four_word_combination = []

#        for word in text_list:

#            four_word_combination.append(word)
            
#            if len(four_word_combination) == 2* window:
#                line = " ".join(four_word_combination)
#                four_word_combination = []
#                f.write(line + '\n')
             

#accuracy_test_set = w2v_model.wv.accuracy('test_data_window.txt')
#n_correct = len(accuracy_test_set[1]['correct'])

#n_incorrect = len(accuracy_test_set[1]['incorrect'])

#n_correct / (n_correct + n_incorrect)






# save model
w2v_model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)