# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:43:30 2021

@author: flori
"""
import pandas as pd
from nltk.util import ngrams
from collections import Counter
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt 


def count_n_unique_top_words(text_str, n_top_words):
    word_tokens = word_tokenize(text_str)
    n_unique_words = len(list(set(word_tokens)))
    
    n_gram_text = ngrams(word_tokens, 1)
    word_counter = Counter(n_gram_text)
    top_words_text = word_counter.most_common(n_top_words)
    
    return n_unique_words, top_words_text

def concat_texts(type_text, list_df):
    
    list_texts = []
    
    for df in list_df:
        for text in df[type_text]:
            if text is None:
                continue
            else:
                
                list_texts.append(text)
    
    concatenated_texts = ' '.join(list_texts)
    
    return concatenated_texts

journals = ['journalofmarketing',
            'journalofmarketingresearch',
            'journalofconsumerresearch',
            'journalofconsumerpsych',
            'journalacademyofmarketingscience']


####### Raw 

list_journals_raw = []

for journal in journals:
    

    # load scraped data
    df_raw = pd.read_parquet('../Data/Scraped/' + journal + '_data_lim.gzip')
    list_journals_raw.append(df_raw)
    
    
df_journals_raw = pd.concat(list_journals_raw)



n_articles_raw = len(df_journals_raw.index)

####### Step 1: Remove in-text references, sub-titles, things inbetween brackets (basically everything for BERT) 


list_journals_BERT = []

for journal in journals:
    

    # load scraped data
    df_BERT = pd.read_parquet('../Data/clean/' + journal + '_BERT.gzip')
    list_journals_BERT.append(df_BERT)
    

# concat all together, get n of articles    
all_body_BERT = concat_texts('body', list_journals_BERT)
df_journals_BERT= pd.concat(list_journals_BERT)
n_articles_BERT = len(df_journals_BERT.index)
n_articles_BERT

# get top words for BERT    
n_unique_words_BERT, top_words_body_BERT = count_n_unique_top_words(all_body_BERT, 10)
n_unique_words_BERT
[word[0] for word in top_words_body_BERT]


######## Step 2: remove stopwords, non-alphabetic, single words

list_journals_step2 = []

for journal in journals:
    
    # load scraped data
    df_step2 = pd.read_parquet('../Data/clean/' + journal + '_allWords_merged.gzip')
    list_journals_step2.append(df_step2)
    

# concat all together, get n of articles    
all_body_step2 = concat_texts('body', list_journals_step2)
df_journals_step2= pd.concat(list_journals_step2)
n_articles_step2 = len(df_journals_step2.index)




# get top words for BERT    
n_unique_words_step2, top_words_body_step2 = count_n_unique_top_words(all_body_step2, 10)
n_unique_words_step2
top_words_body_step2

######## Step 3: only adjectives and nouns 

# load scraped data
df_full = pd.read_parquet('../Data/clean/all_journals_adject_nouns_merged_withLemmatized.gzip')
    

# concat all together, get n of articles    
all_body_step3 = ' '.join(df_full['body'])
n_articles_step3 = len(df_full.index)
n_articles_step3

# get top words for BERT    
n_unique_words_step3, top_words_body_step3 = count_n_unique_top_words(all_body_step3, 10)
n_unique_words_step3
top_words_body_step3

######## Step 4: lemmatized

all_body_step4 = ' '.join(df_full['body_lemmatized'])
n_unique_words_step4, top_words_body_step4 = count_n_unique_top_words(all_body_step4, 10)
n_unique_words_step4
top_words_body_step4



##### quick plot on comparison
df_n_per_year_before_clean = df_journals_step2[['Year', 'DOI']].groupby('Year').count()
df_n_per_year_after_clean = df_full[['year', 'DOI']].groupby('year').count()
df_n_per_year = pd.concat([df_n_per_year_before_clean,df_n_per_year_after_clean], axis = 1)
df_n_per_year.columns = ['before', 'after']

plt.bar(df_n_per_year.index, df_n_per_year['before'],  label = 'Before Cleaning', color = 'red')
plt.bar(df_n_per_year.index, df_n_per_year['after'], label = 'After Cleaning',  color = 'grey')
plt.legend(loc = 'upper left')
plt.savefig('over_time_sample.png', dpi=200)

