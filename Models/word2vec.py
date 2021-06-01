import multiprocessing
from gensim.models import Word2Vec
from nltk import tokenize
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib as plt
import numpy as np


df = pd.read_parquet('../Data/clean/data_journals_merged.gzip')
body_texts = df['body'].tolist()
df.index = [i for i in range(1, len(df.index)+1)]
df.index

texts = [row.split() for row in df['body']]

#random_indeces = list(np.random.permutation(len(texts)))
#train_size = int(round(len(texts) * 0.7,0))

#train_indeces = random_indeces[:train_size]
#test_indeces = random_indeces[train_size:]

#train_texts = [texts[i] for i in train_indeces]
#test_texts = [texts[j] for j in test_indeces]

cores = multiprocessing.cpu_count()


# window: 5, why? what other words (of any type) are used in related discussions? Smaller windows tend to capture more about word itself: what other words are functionally similar?
# https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf
window = 5
# size = dimensionality of the vector; needs to be a lot lower than the vocabulary
size = 100 # consider 300

# alpha: learning rate
alpha = 0.025 

# based on mikolov et al; negative sampling
negative = 5

# original word2vec paper
ns_exponent = 0.75

w2v_model = Word2Vec(texts,
                     window=window,
                     size=size,
                     alpha = alpha,
                     negative = negative,
                     workers=cores-1)




# summarize model
# summarize vocabulary
vocab = list(w2v_model.wv.vocab)
len(vocab)

w2v_model.most_similar('retail')
w2v_model.most_similar('advertisement')
w2v_model.most_similar('effect')


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



list_doc_representation = [document_vector_mean(doc, w2v_model) for doc in texts ]
len(list_doc_representation)

list_doc_representation
names = ['mean_dim_' + str(i) for i in range(1,size + 1)]
df_doc_representation = pd.DataFrame(list_doc_representation)
df_doc_representation.columns = names
df_doc_representation['num_ref']= df['num_ref']
df_doc_representation


df_doc_representation.to_csv('../data/representations/word2vec_doc_representation.csv')















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