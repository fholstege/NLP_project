
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
df_journals_merged_all = pd.read_parquet('../data/clean/all_journals_allWords_merged.gzip')


body_list = list(flatten( [[word for word in document.split()] for document in df_journals_merged['body_lemmatized'].tolist()]))
abstract_list = list(flatten( [[word for word in document.split()] for document in df_journals_merged['abstract_lemmatized'].tolist()]))
title_list = list(flatten( [[word for word in document.split()] for document in df_journals_merged['title_lemmatized'].tolist()]))

body_list_all = list(flatten( [[word for word in document.split()] for document in df_journals_merged_all['body_lemmatized'].tolist()]))
abstract_list_all = list(flatten( [[word for word in document.split()] for document in df_journals_merged_all['abstract_lemmatized'].tolist()]))
title_list_all = list(flatten( [[word for word in document.split()] for document in df_journals_merged_all['title_lemmatized'].tolist()]))


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


###### adjectives, nouns

# top uni and bigrams - body of text
top_words_body = get_top_n_gram(body_list, 1, 20)
top_two_words_body =  get_top_n_gram(body_list, 2, 20, remove_double = True)

# top uni and bigrams - abstract
top_words_abstract =  get_top_n_gram(abstract_list, 1, 20)
top_two_words_abstract =  get_top_n_gram(abstract_list, 2, 20, remove_double = True)

# top uni and bigrams - titles
top_words_title = get_top_n_gram(title_list, 1, 20)
top_two_words_title =  get_top_n_gram(title_list, 2, 20, remove_double = True)


create_plot_n_gram(top_words_body, 'blue', save = True, figname = 'unigram_body_plot')
create_plot_n_gram(top_two_words_body, 'red', save = True, figname = 'bigram_body_plot')
create_plot_n_gram(top_words_abstract, 'green', save = True, figname = 'unigram_abstract_plot')
create_plot_n_gram(top_two_words_abstract, 'purple', save = True, figname = 'bigram_abstract_plot')
create_plot_n_gram(top_words_title, 'orange', save = True, figname = 'unigram_title_plot')
create_plot_n_gram(top_two_words_title, 'pink', save = True, figname = 'bigram_title_plot')




###### all words

# top uni and bigrams - body of text
top_words_body_all = get_top_n_gram(body_list_all, 1, 20)
top_two_words_body_all =  get_top_n_gram(body_list_all, 2, 20, remove_double = True)

# top uni and bigrams - abstract
top_words_abstract_all =  get_top_n_gram(abstract_list_all, 1, 20)
top_two_words_abstract_all =  get_top_n_gram(abstract_list_all, 2, 20, remove_double = True)

# top uni and bigrams - titles
top_words_title_all = get_top_n_gram(title_list_all, 1, 20)
top_two_words_title_all =  get_top_n_gram(title_list_all, 2, 20, remove_double = True)



create_plot_n_gram(top_words_body_all, 'blue', save = True, figname = 'unigram_body_plot_all')
create_plot_n_gram(top_two_words_body_all, 'red', save = True, figname = 'bigram_body_plot_all')
create_plot_n_gram(top_words_abstract_all, 'green', save = True, figname = 'unigram_abstract_plot_all')
create_plot_n_gram(top_two_words_abstract_all, 'purple', save = True, figname = 'bigram_abstract_plot_all')
create_plot_n_gram(top_words_title_all, 'orange', save = True, figname = 'unigram_title_plot_all')
create_plot_n_gram(top_two_words_title_all, 'pink', save = True, figname = 'bigram_title_plot_all')



