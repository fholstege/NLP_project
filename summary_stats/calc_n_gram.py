
from collections import Counter
from nltk import word_tokenize
from nltk.util import ngrams
import pandas as pd
from helpfunctions_summary_stats import take
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer 
from pandas.core.common import flatten
 
# read the zipped files
df_jom_merged = pd.read_parquet('../Data/clean/journalofmarketing_merged.gzip')
df_jomr_merged = pd.read_parquet('../Data/clean/journalofmarketingresearch_merged.gzip')
df_jcr_merged = pd.read_parquet('../Data/clean/journalofconsumerresearch_merged.gzip')
df_jcp_merged = pd.read_parquet('../Data/clean/journalofconsumerpsych_merged.gzip')
df_jam_merged = pd.read_parquet('../Data/clean/journalacademyofmarketingscience_merged.gzip')

# add journal titles
df_jom_merged['Journal'] = 'Journal of Marketing' 
df_jomr_merged['Journal'] = 'Journal of Marketing Research' 
df_jcr_merged['Journal'] = 'Journal of Consumer Research' 
df_jcp_merged['Journal'] = 'Journal of Consumer Psychology' 
df_jam_merged['Journal'] = 'Journal Academy of Marketing Science' 

# get dataframes together
frames_journals = [df_jom_merged, df_jomr_merged, df_jcr_merged, df_jcp_merged, df_jam_merged]

# merge the journals in one dataframe, vertically
df_journals_merged = pd.concat(frames_journals)


# init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# create list of words for each document -body
body_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['body'].tolist()]
body_flat_list = list(flatten(body_list))  

# create list of words for each document -abstract
abstract_list =  [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['abstract'].tolist()]
abstract_flat_list =  list(flatten(abstract_list))  

# create list of words for each document -titles
title_list =  [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['title'].tolist()]
title_flat_list =  list(flatten(title_list)) 

 


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


def get_top_n_gram(text_list, n, top_n):
    
    n_gram_text = ngrams(text_list, n)
    word_counter = Counter(n_gram_text)
    top_words_text = word_counter.most_common(top_n)
    
    return top_words_text

def create_plot_n_gram(top_words, color):
    
    list_words, list_words_counts = extract_values_counter(top_words) 
    
    list_words.reverse()
    list_words_counts.reverse()
    
    # plot of n-gram
    plt.barh(list_words, width = list_words_counts, color = color)
    plt.tick_params(axis='both', which='major', length=5, labelsize = 'small')
    plt.show()


# top uni and bigrams - body of text
top_words_body = get_top_n_gram(body_flat_list, 1, 20)
top_two_words_body =  get_top_n_gram(body_flat_list, 2, 20)

# top uni and bigrams - abstract
top_words_abstract =  get_top_n_gram(abstract_flat_list, 1, 20)
top_two_words_abstract =  get_top_n_gram(abstract_flat_list, 2, 20)

# top uni and bigrams - titles
top_words_title = get_top_n_gram(title_flat_list, 1, 20)
top_two_words_title =  get_top_n_gram(title_flat_list, 2, 20)


create_plot_n_gram(top_words_body, 'blue')
create_plot_n_gram(top_two_words_body, 'red')
create_plot_n_gram(top_words_abstract, 'green')
create_plot_n_gram(top_two_words_abstract, 'purple')
create_plot_n_gram(top_words_title, 'orange')
create_plot_n_gram(top_two_words_title, 'pink')

