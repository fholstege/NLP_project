
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

# create list of words for each document
texts_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df_journals_merged['body'].tolist()]
text_flat_list = list(flatten(texts_list))    

uni_gram = ngrams(text_flat_list,1)
bi_gram = ngrams(text_flat_list,2)

word_counter = Counter(uni_gram)
top_words = word_counter.most_common(20)

bigram_counter = Counter(bi_gram)
top_two_words = bigram_counter.most_common(20)


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

# get list of words, and for two words
list_words, list_word_counts = extract_values_counter(top_words)    
list_two_words, list_two_word_counts = extract_values_counter(top_two_words)


list_words.reverse()
list_word_counts.reverse()
list_two_words.reverse()
list_two_word_counts.reverse()

# plot of unigram
plt.barh(list_words, width = list_word_counts)
plt.tick_params(axis='both', which='major', length=5, labelsize = 'small')

# plot of bigrams
plt.barh(list_two_words, width = list_two_word_counts)
plt.tick_params(axis='both', which='major', length=5, labelsize = 'small')




