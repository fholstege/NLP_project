import multiprocessing
from gensim.models import Word2Vec
from nltk import tokenize
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib as plt
import numpy as np
from nltk.stem import WordNetLemmatizer 
from pandas.core.common import flatten
import pandas as pd

# read merged files, get body of text, adapt index

# stack data from all journals
df = pd.read_parquet('../Data/clean/all_journals_adject_nouns_merged.gzip')
df = df.reset_index()
len(df.index)

# split the texts
texts = [row.split() for row in df['body_lemmatized'].str.lower().tolist()]



# create train and test set, save which ones are train and test
random_indeces = list(np.random.permutation(len(texts)))
train_size = int(round(len(texts) * 0.7,0))
train_indeces = random_indeces[:train_size]
test_indeces = random_indeces[train_size:]
train_bool =  [1 if i in train_indeces else 0 for i in range(0,max(random_indeces)+1)]
train_texts = [texts[i] for i in train_indeces]
test_texts = [texts[j] for j in test_indeces]


# get the number of cores
cores = multiprocessing.cpu_count() - 4


# window: 5, why? what other words (of any type) are used in related discussions? Smaller windows tend to capture more about word itself: what other words are functionally similar?
# https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf
window = 5
# size = dimensionality of the vector; needs to be a lot lower than the vocabulary
# mikolov: 300
size_1 = 300 
size_2 = 500

# alpha: learning rate
alpha = 0.025 

# based on mikolov et al; negative sampling
negative = 5

# use the cbow
sg = 0

# train word2vec on training set
w2v_model_300 = Word2Vec(train_texts,
                     window=window,
                     size=size_1,
                     alpha = alpha,
                     negative = negative,
                     sg = sg,
                     workers=cores)
# train word2vec on training set
w2v_model_500 = Word2Vec(train_texts,
                     window=window,
                     size=size_2,
                     alpha = alpha,
                     negative = negative,
                     sg = sg,
                     workers=cores)




# summarize for couple of common words which words are similar
def get_table_similar(words, model):
    
    dict_words = {element:0 for element in words}
    
    for word in words:
        similar_words = model.most_similar(word)
        
        list_similar_words = []
        
        for similar_word in similar_words:
            list_similar_words.append(similar_word[0])
        
        dict_words[word] = list_similar_words
    
    return dict_words


# get table of common words 
check_words = ['effect', 'product', 'research', 'brand']

# 300 
df_common_words_300 = pd.DataFrame(get_table_similar(check_words, w2v_model_300))
df_common_words_300.index = list(range(1,10+1,1))

# 500
df_common_words_500 = pd.DataFrame(get_table_similar(check_words, w2v_model_500))
df_common_words_500.index = list(range(1,10+1,1))
print(df_common_words_500.to_latex())



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



list_doc_representation_mean_300 = [document_vector_mean(doc, w2v_model_300) for doc in texts ]
list_doc_representation_max_300 = [document_vector_max(doc, w2v_model_300) for doc in texts ]
list_doc_representation_min_300 = [document_vector_min(doc, w2v_model_300) for doc in texts ]


list_doc_representation_mean_500 = [document_vector_mean(doc, w2v_model_500) for doc in texts ]
list_doc_representation_max_500 = [document_vector_max(doc, w2v_model_500) for doc in texts ]
list_doc_representation_min_500 = [document_vector_min(doc, w2v_model_500) for doc in texts ]



names_mean_300 = ['mean_dim_' + str(i) for i in range(1,size_1 + 1)]
names_max_300 = ['max_dim_' + str(i) for i in range(1,size_1 + 1)]
names_min_300 = ['min_dim_' + str(i) for i in range(1,size_1 + 1)]



names_mean_500 = ['mean_dim_' + str(i) for i in range(1,size_2 + 1)]
names_max_500 = ['max_dim_' + str(i) for i in range(1,size_2 + 1)]
names_min_500 = ['min_dim_' + str(i) for i in range(1,size_2 + 1)]


def create_df_representation(list_representation, names):
    df_representation = pd.DataFrame(list_representation)
    df_representation.columns = names
    df_representation['citations'] = df['citations']
    df_representation['train_set']  = train_bool
    df_representation['year']  = df['year']
    df_representation['DOI'] = df['DOI']
    
    return df_representation


    
df_doc_representation_mean_300 = create_df_representation(list_doc_representation_mean_300, names_mean_300)
df_doc_representation_max_300 = create_df_representation(list_doc_representation_max_300, names_max_300)
df_doc_representation_min_300 = create_df_representation(list_doc_representation_min_300, names_min_300)

df_doc_representation_mean_500 = create_df_representation(list_doc_representation_mean_500, names_mean_500)
df_doc_representation_max_500 = create_df_representation(list_doc_representation_max_500, names_max_500)
df_doc_representation_min_500 = create_df_representation(list_doc_representation_min_500, names_min_500)




df_doc_representation_mean_300.to_parquet('../data/representations/word2vec_doc_representation_300_mean.gzip',compression='gzip')
df_doc_representation_max_300.to_parquet('../data/representations/word2vec_doc_representation_300_max.gzip', compression='gzip')
df_doc_representation_min_300.to_parquet('../data/representations/word2vec_doc_representation_300_min.gzip', compression='gzip')


df_doc_representation_mean_500.to_parquet('../data/representations/word2vec_doc_representation_500_mean.gzip',compression='gzip')
df_doc_representation_max_500.to_parquet('../data/representations/word2vec_doc_representation_500_max.gzip', compression='gzip')
df_doc_representation_min_500.to_parquet('../data/representations/word2vec_doc_representation_500_min.gzip', compression='gzip')














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