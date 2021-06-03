from remove_functions import standard_cleaner, html_cleaner_jam, html_cleaner_jcp, html_cleaner_jcr, xml_cleaner_jom_jomr, remove_NUM_tag, remove_stopwords_non_alpha_single_words
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize 
from nltk.util import ngrams
from collections import Counter

# enable printed progress for apply functions
from tqdm import tqdm
tqdm.pandas()


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

journal = journals[0]


for journal in journals:
    
    print('Begin construction of BERT data for ' + journal)

    # load scraped data
    df = pd.read_parquet('../Data/Scraped/' + journal + '_data_lim.gzip')

    # journal specific cleaning
    if journal in ['journalofmarketing', 'journalofmarketingresearch']:
        df['body'] = df['body'].progress_apply(xml_cleaner_jom_jomr)
    elif journal == 'journalofconsumerresearch':
        df['body'] = df['body'].progress_apply(html_cleaner_jcr)
    elif journal == 'journalofconsumerpsych':
        df['body'] = df['body'].progress_apply(html_cleaner_jcp)
    elif journal == 'journalacademyofmarketingscience':
        df['body'] = df['body'].progress_apply(html_cleaner_jam) 
        
    # standard cleaning
    df['body'] = df['body'].progress_apply(standard_cleaner)
    
    # HERE IT FAILS THE STANDARD CLEANING
    # replicate: run above code until and including journal = journals[0] 
    # then run df = pd.read_parquet('../Data/Scraped/' + journal + '_data_lim.gzip')
    # then df['body'] = df['body'].progress_apply(xml_cleaner_jom_jomr)
    # then the standard cleaning above
    
    # proof that it works not using apply (both for non-NA and NA docs)
    test = standard_cleaner(df['body'][42])
    standard_cleaner(df.loc[df['body'] == 'NA', 'body'][462])
    
    # check type
    df['body'].dtype
    type(df['body'][0])
    
    
    
    # save BERT type corpus
    df.to_parquet("../Data/clean/" + journal + "_BERT.gzip", compression='gzip')
    
    print('Saved ' + journal + '_BERT')

#######################################################################################

## Part 2 - allWords data: remove single words, non-alphabetic words, and stopwords

    print('Begin construction of allWords data for ' + journal)

    # remove numbers based on NUM tag
    df['body'] = df['body'].apply(remove_NUM_tag)

    # lower-case everything (cannot do before because NUM should not be removed for BERT)
    df['body'] = df['body'].str.lower()
    df['title'] = df['title'].str.lower()
    df['abstract'] = df['abstract'].str.lower()

    # remove stopwords and non-alpha words
    df['body'] = df['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['abstract'] = df['abstract'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['title'] = df['title'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)

    # data with all types of words
    df.to_parquet("../Data/clean/" + journal + "_allWords.gzip", compression='gzip')

    print('Saved ' + journal + '_allWords')

#######################################################################################

## Part 3 - adject_nouns data: only keep nouns and adjectives based on POS tagging

    print('Begin construction of adject_nouns data for ' + journal)

    # base POS tagging on BERT dataset as it has better sentence structure
    df = pd.read_parquet('../Data/clean/' + journal + '_BERT.gzip')

    # remove numbers based on NUM tag
    df['body'] = df['body'].apply(remove_NUM_tag)

    # remove all words except adjectives and nouns 
    df['body'] = df['body'].apply(keep_nouns_adjectives)
    df['abstract'] = df['abstract'].apply(keep_nouns_adjectives)
    df['title'] = df['title'].apply(keep_nouns_adjectives)

    # remove stopwords and non-alpha words
    df['body'] = df['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['abstract'] = df['abstract'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
    df['title'] = df['title'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)

    # to gzip 
    df.to_parquet('../Data/clean/' + journal + '_adject_nouns.gzip', compression='gzip')

    print('Saved ' + journal + '_adject_nouns')

#######################################################################################

# additional cleaning
# deal with either/both and similar word compositions?