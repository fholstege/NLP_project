from remove_functions import standard_cleaner, html_cleaner_jam, html_cleaner_jcp, html_cleaner_jcr, xml_cleaner_jom_jomr, remove_NUM_tag, remove_stopwords_non_alpha_single_words, keep_nouns_adjectives
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize 
from nltk.util import ngrams
from collections import Counter
from nltk.stem import WordNetLemmatizer

# enable printed progress for apply functions
from tqdm import tqdm
tqdm.pandas()



lemmatizer = WordNetLemmatizer()


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
            list_texts.append(text)
    
    concatenated_texts = ' '.join(list_texts)
    
    return concatenated_texts

#######################################################################################       

# STOPWORDS

# create stopword list
stopwords_nltk = pd.read_csv('../Data/stopwords/NLTK.txt', sep = "\n", header = None)[0].tolist()
stopwords_technical_add = pd.read_csv('../Data/stopwords/Technical_add.txt', sep = "\n", header = None)[0].tolist()
stopwords_TN_add = pd.read_csv('../Data/stopwords/TN_additional.txt', sep = "\n", header = None)[0].tolist()
stopwords_USTPO = pd.read_csv('../Data/stopwords/USTPO.txt', sep = "\n", header = None)[0].tolist()
stopwords_own = pd.read_csv('../Data/stopwords/own_stopwords.txt', sep = "\n", header = None)[0].tolist()

# combine to one stopword list
stopwords_final = stopwords_nltk + stopwords_technical_add + stopwords_TN_add + stopwords_USTPO + stopwords_own

#######################################################################################

### Part 1: raw data cleaning

journals = ['journalofmarketing',
            'journalofmarketingresearch',
            'journalofconsumerresearch',
            'journalofconsumerpsych',
            'journalacademyofmarketingscience']

for journal in journals:
    
    print('\nBegin construction of BERT data for ' + journal)

    # load scraped data
    df = pd.read_parquet('../Data/Scraped/' + journal + '_data_lim.gzip')
    
    # remove columns not of interest
    df = df[['DOI', 'title', 'abstract', 'body']]

    # journal specific cleaning
    if journal in ['journalofmarketing', 'journalofmarketingresearch']:
        df['body'] = df['body'].progress_apply(xml_cleaner_jom_jomr)
    elif journal == 'journalofconsumerresearch':
        df['body'] = df['body'].progress_apply(html_cleaner_jcr)
    elif journal == 'journalofconsumerpsych':
        df['body'] = df['body'].progress_apply(html_cleaner_jcp)
    elif journal == 'journalacademyofmarketingscience':
        df['body'] = df['body'].progress_apply(html_cleaner_jam) 
        
    # remove articles with missing body
    print(f"\nRemove {sum(df['body'] == 'NA')} articles")
    df = df.loc[df['body'] != 'NA']

    # standard cleaning
    print('\nDo standard cleaning')
    df['body'] = df['body'].progress_apply(standard_cleaner)
    
    # save BERT type corpus
    df = df.loc[df['body'] != '']
    df.to_parquet("../Data/clean/" + journal + "_BERT.gzip", compression='gzip')
    
    print('\nSaved ' + journal + '_BERT')

#######################################################################################

## Part 2 - allWords data: remove single words, non-alphabetic words, and stopwords

    print('\nBegin construction of allWords data for ' + journal)

    # remove numbers based on NUM tag
    print('\nRemove NUM tags')
    df['body'] = df['body'].apply(remove_NUM_tag)

    # lower-case everything (cannot do before because NUM should not be removed for BERT)
    df['body'] = df['body'].str.lower()
    df['title'] = df['title'].str.lower()
    df['abstract'] = df['abstract'].str.lower()

    # remove stopwords and non-alpha words
    print('\nRemove stopwords')
    df['body'] = df['body'].progress_apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['abstract'] = df['abstract'].progress_apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['title'] = df['title'].progress_apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)

    # save data with all types of words
    df = df.loc[df['body'] != '']
    df.to_parquet("../Data/clean/" + journal + "_allWords.gzip", compression='gzip')

    print('\nSaved ' + journal + '_allWords')

#######################################################################################

## Part 3 - adject_nouns data: only keep nouns and adjectives based on POS tagging

    print('\nBegin construction of adject_nouns data for ' + journal)

    # base POS tagging on BERT dataset as it has better sentence structure
    df = pd.read_parquet('../Data/clean/' + journal + '_BERT.gzip')

    # remove numbers based on NUM tag
    print('\nRemove NUM tags')
    df['body'] = df['body'].apply(remove_NUM_tag)

    # remove all words except adjectives and nouns 
    print('\nKeep only adjectives and nouns')
    df['body'] = df['body'].progress_apply(keep_nouns_adjectives)
    df['abstract'] = df['abstract'].astype(str).progress_apply(keep_nouns_adjectives)
    df['title'] = df['title'].astype(str).progress_apply(keep_nouns_adjectives)

    # remove stopwords and non-alpha words
    print('\nRemove stopwords')
    df['body'] = df['body'].progress_apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['abstract'] = df['abstract'].progress_apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['title'] = df['title'].progress_apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)

    # save to gzip 
    df = df.loc[df['body'] != '']
    df.to_parquet('../Data/clean/' + journal + '_adject_nouns.gzip', compression='gzip')

    print('\nSaved ' + journal + '_adject_nouns')

#######################################################################################

# MERGE JOURNALS INTO ONE DATASET PER TYPE

data_types = ['_BERT', '_allWords', '_adject_nouns']

# define lemmatizer
lemmatizer = WordNetLemmatizer()

for data_type in data_types:
    # stack data from all journals
    df = pd.read_parquet('../data/clean/' + journals[0] + data_type + '.gzip')
    
    for journal in journals[1:len(journals)]:
        df = pd.concat([df, pd.read_parquet('../data/clean/' + journal + data_type + '.gzip')]) 
    
    df = df.reset_index()
    
    # no lemmatization for BERT type data
    if data_type == '_allWords' or data_type == '_adject_nouns':
        # lemmatization - list of list
        body_lemmatized_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df['body'].tolist()]
        abstract_lemmatized_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df['abstract'].tolist()]
        title_lemmatized_list = [[lemmatizer.lemmatize(word) for word in document.split()] for document in df['abstract'].tolist()]
        
        # turn to strings
        df['body_lemmatized'] = [' '.join(body) for body in body_lemmatized_list]
        df['abstract_lemmatized'] = [' '.join(abstract) for abstract in abstract_lemmatized_list]
        df['title_lemmatized'] = [' '.join(title) for title in title_lemmatized_list]
    
    if data_type == '_BERT':
        df['body_lemmatized'] = 'NA'
        df['abstract_lemmatized'] = 'NA'
        df['title_lemmatized'] = 'NA'
        
    # save new dataframe
    df.to_parquet('../data/clean/all_journals' + data_type + '.gzip')
    print('all_journals' + data_type + ' saved')