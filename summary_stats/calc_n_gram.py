
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd
from helpfunctions_summary_stats import take
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
from pandas.core.common import flatten
 
# merge the journals in one dataframe, vertically
df_journals_merged = pd.read_parquet('../data/clean/all_journals_adject_nouns_merged.gzip')

# init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# create list of words for each document -body
body_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['body'].tolist()]
body_lemmatized_list = list(flatten(body_list))  
body_lemmatized_list_str = [' '.join(body) for body in body_list ]

# create list of words for each document -abstract
abstract_list =  [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['abstract'].tolist()]
abstract_lemmatized_list =  list(flatten(abstract_list))  
abstract_lemmatized_list_str = [' '.join(abstract) for abstract in abstract_list ]

# create list of words for each document -titles
title_list =  [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['title'].tolist()]
title_lemmatized_list =  list(flatten(title_list)) 
title_lemmatized_list_str = [' '.join(title) for title in title_list ]


# add lemmatized versions backs
df_journals_merged['body_lemmatized'] = body_lemmatized_list_str
df_journals_merged['abstract_lemmatized'] = abstract_lemmatized_list_str
df_journals_merged['title_lemmatized'] = title_lemmatized_list_str
df_journals_merged.to_parquet("../Data/clean/all_journals_adject_nouns_merged_withLemmatized.gzip", compression='gzip')

df_journals_merged = pd.read_parquet('../data/clean/all_journals_adject_nouns_merged_withLemmatized.gzip')


def convertTuple(tup):
    str =  ' '.join(tup)
    return str

def extract_values_counter(Counter_obj, convert_tuple = True):
    
    dict_obj = dict(Counter_obj)
    list_keys = list(dict_obj.keys())
    list_values = list(dict_obj.values())
    
    if convert_tuple:
        list_keys =  [convertTuple(i) for i in list_keys]
        
    return list_keys, list_values 


def get_top_n_gram(text_list, n, top_n, remove_double = False):
    
    n_gram_text = ngrams(text_list, n)
    word_counter = Counter(n_gram_text)
        
    
    
    if remove_double:
        keys_to_remove = []
                
        for word_combination in word_counter.keys():
            
            if word_combination[0] == word_combination[1]:
               keys_to_remove.append(word_combination)
        
        for word_combi in keys_to_remove:
            if word_combi in word_counter:
                del word_counter[word_combi]
        
        
    top_words_text = word_counter.most_common(top_n)
    
    return top_words_text

def create_plot_n_gram(top_words, color, save= False, dpi = 200, figname = None):
    
    list_words, list_words_counts = extract_values_counter(top_words) 
    
    list_words.reverse()
    list_words_counts.reverse()
    
    # plot of n-gram
    plt.barh(list_words, width = list_words_counts, color = color)
    plt.tick_params(axis='both', which='major', length=5, labelsize = 'small')
    
    if save:
        plt.tight_layout()
        plt.savefig(figname, dpi=dpi)

    plt.show()


# top uni and bigrams - body of text
top_words_body = get_top_n_gram(df_journals_merged['body_lemmatized'].tolist(), 1, 20)
top_two_words_body =  get_top_n_gram(df_journals_merged['body_lemmatized'].tolist(), 2, 20, remove_double = True)

# top uni and bigrams - abstract
top_words_abstract =  get_top_n_gram(df_journals_merged['abstract_lemmatized'], 1, 20)
top_two_words_abstract =  get_top_n_gram(df_journals_merged['abstract_lemmatized'], 2, 20, remove_double = True)

# top uni and bigrams - titles
top_words_title = get_top_n_gram(df_journals_merged['title_lemmatized'], 1, 20)
top_two_words_title =  get_top_n_gram(df_journals_merged['title_lemmatized'], 2, 20, remove_double = True)


create_plot_n_gram(top_words_body, 'blue', save = True, figname = 'unigram_body_plot')
create_plot_n_gram(top_two_words_body, 'red', save = True, figname = 'bigram_body_plot')
create_plot_n_gram(top_words_abstract, 'green', save = True, figname = 'unigram_abstract_plot')
create_plot_n_gram(top_two_words_abstract, 'purple', save = True, figname = 'bigram_abstract_plot')
create_plot_n_gram(top_words_title, 'orange', save = True, figname = 'unigram_title_plot')
create_plot_n_gram(top_two_words_title, 'pink', save = True, figname = 'bigram_title_plot')

