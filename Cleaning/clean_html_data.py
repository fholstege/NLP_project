from remove_functions import remove_in_text_references_text, remove_stopwords_non_alpha_single_words,remove_titles_sage, remove_in_text_references_text, keep_nouns_adjectives
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize 
from nltk.util import ngrams
from collections import Counter


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
            


### Part 1: raw data 


# read the zipped files
df_jom = pd.read_parquet('../Data/Scraped/journalofmarketing_data_lim.gzip')
df_jomr = pd.read_parquet('../Data/Scraped/journalofmarketingresearch_data_lim.gzip')
df_jcr = pd.read_parquet('../Data/Scraped/journalofconsumerresearch_data_lim.gzip')
df_jcp = pd.read_parquet('../Data/Scraped/journalofconsumerpsych_data_lim.gzip')
df_jam = pd.read_parquet('../Data/Scraped/journalacademyofmarketingscience_data_lim.gzip')




## Part 2: remove titles and in-text references


# remove titles from html
df_jom['body'] = df_jom['body'].apply(remove_titles_sage)
df_jomr['body'] = df_jomr['body'].apply(remove_titles_sage)


# remove in text references
df_jom['body'] = df_jom['body'].apply(remove_in_text_references_text, publisher = 'SAGE')
df_jomr['body'] = df_jomr['body'].apply(remove_in_text_references_text, publisher = 'SAGE')
df_jcr['body'] = df_jcr['body'].apply(remove_in_text_references_text, publisher = 'OUP')
df_jcp['body'] = df_jcp['body'].apply(remove_in_text_references_text, publisher = 'Wiley')
df_jam['body'] = df_jam['body'].apply(remove_in_text_references_text, publisher = 'springer')



 ## count summary statistics for  part 2
df_journals_list_part2 = [df_jom, df_jomr,df_jcr, df_jcp, df_jam ]

all_text_body_part2 = concat_texts('body',df_journals_list_part2 )
count_n_unique_top_words(all_text_body_part2, 10)

all_text_abstract_part2 = concat_texts('abstract',df_journals_list_part2 )
count_n_unique_top_words(all_text_abstract_part2, 10)

all_text_titles_part2 = concat_texts('title',df_journals_list_part2 )
count_n_unique_top_words(all_text_titles_part2, 10)

## Turn all to lower

# turn all to lower
df_jom['body'] = df_jom['body'].str.lower()
df_jomr['body'] = df_jomr['body'].str.lower()
df_jcr['body'] = df_jcr['body'].str.lower()
df_jcp['body'] = df_jcp['body'].str.lower()
df_jam['body'] = df_jam['body'].str.lower()

# do the same for abstracts
df_jom['abstract'] = df_jom['abstract'].str.lower()
df_jomr['abstract'] = df_jomr['abstract'].str.lower()
df_jcr['abstract'] = df_jcr['abstract'].str.lower()
df_jcp['abstract'] = df_jcp['abstract'].str.lower()
df_jam['abstract'] = df_jam['abstract'].str.lower()

# do the same for titles
df_jom['title'] = df_jom['title'].str.lower()
df_jomr['title'] = df_jomr['title'].str.lower()
df_jcr['title'] = df_jcr['title'].str.lower()
df_jcp['title'] = df_jcp['title'].str.lower()
df_jam['title'] = df_jam['title'].str.lower()


# BERT data created after step 2
df_jom.to_parquet("../Data/clean/journalofmarketing_BERT.gzip", compression='gzip')
df_jomr.to_parquet("../Data/clean/journalofmarketingresearch_BERT.gzip", compression='gzip')
df_jcr.to_parquet("../Data/clean/journalofconsumerresearch_BERT.gzip", compression='gzip')
df_jcp.to_parquet("../Data/clean/journalofconsumerpsych_BERT.gzip", compression='gzip')
df_jam.to_parquet("../Data/clean/journalacademyofmarketingscience_BERT.gzip", compression='gzip')


## Part 3: remove single words, non-alphabetic words, and stopwords


# create stopword list
stopwords_nltk = pd.read_csv('../Data/stopwords/NLTK.txt', sep = "\n", header = None)[0].tolist()
stopwords_technical_add = pd.read_csv('../Data/stopwords/Technical_add.txt', sep = "\n", header = None)[0].tolist()
stopwords_TN_add = pd.read_csv('../Data/stopwords/TN_additional.txt', sep = "\n", header = None)[0].tolist()
stopwords_USTPO = pd.read_csv('../Data/stopwords/USTPO.txt', sep = "\n", header = None)[0].tolist()
stopwords_own = pd.read_csv('../Data/stopwords/own_stopwords.txt', sep = "\n", header = None)[0].tolist()

# combine to one stopword list
stopwords_final = stopwords_nltk + stopwords_technical_add + stopwords_TN_add + stopwords_USTPO + stopwords_own


# remove stopwords and non-alpha words
df_jom['body'] = df_jom['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jomr['body'] = df_jomr['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcr['body'] = df_jcr['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcp['body'] = df_jcp['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jam['body'] = df_jam['body'].apply(remove_stopwords_non_alpha_single_words,stopword_list = stopwords_final)


# do the same for abstracts
df_jom['abstract'] = df_jom['abstract'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jomr['abstract'] = df_jomr['abstract'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcr['abstract'] = df_jcr['abstract'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcp['abstract'] = df_jcp['abstract'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jam['abstract'] = df_jam['abstract'].apply(remove_stopwords_non_alpha_single_words,stopword_list = stopwords_final)


# do the same for titles
df_jom['title'] = df_jom['title'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jomr['title'] = df_jomr['title'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcr['title'] = df_jcr['title'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcp['title'] = df_jcp['title'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jam['title'] = df_jam['title'].apply(remove_stopwords_non_alpha_single_words,stopword_list = stopwords_final)


 ## count summary statistics for part 3
df_journals_list_part3 = [df_jom, df_jomr,df_jcr, df_jcp, df_jam ]

all_text_body_part3 = concat_texts('body',df_journals_list_part3 )
count_n_unique_top_words(all_text_body_part3, 10)

all_text_abstract_part3 = concat_texts('abstract',df_journals_list_part3 )
count_n_unique_top_words(all_text_abstract_part3, 10)

all_text_titles_part3 = concat_texts('title',df_journals_list_part3 )
count_n_unique_top_words(all_text_titles_part3, 10)



# data with all types of words
df_jom.to_parquet("../Data/clean/journalofmarketing_allWords.gzip", compression='gzip')
df_jomr.to_parquet("../Data/clean/journalofmarketingresearch_allWords.gzip", compression='gzip')
df_jcr.to_parquet("../Data/clean/journalofconsumerresearch_allWords.gzip", compression='gzip')
df_jcp.to_parquet("../Data/clean/journalofconsumerpsych_allWords.gzip", compression='gzip')
df_jam.to_parquet("../Data/clean/journalacademyofmarketingscience_allWords.gzip", compression='gzip')


## Part 4: only keep nouns and adjectives 


# remove all words except adjectives and nouns 
df_jom['body'] = df_jom['body'].apply(keep_nouns_adjectives)
df_jomr['body'] = df_jomr['body'].apply(keep_nouns_adjectives)
df_jcr['body'] = df_jcr['body'].apply(keep_nouns_adjectives)
df_jcp['body'] = df_jcp['body'].apply(keep_nouns_adjectives)
df_jam['body'] = df_jam['body'].apply(keep_nouns_adjectives)


# do the same for abstracts
df_jom['abstract'] = df_jom['abstract'].apply(keep_nouns_adjectives)
df_jomr['abstract'] = df_jomr['abstract'].apply(keep_nouns_adjectives)
df_jcr['abstract'] = df_jcr['abstract'].apply(keep_nouns_adjectives)
df_jcp['abstract'] = df_jcp['abstract'].apply(keep_nouns_adjectives)
df_jam['abstract'] = df_jam['abstract'].apply(keep_nouns_adjectives)


# do the same for titles
df_jom['title'] = df_jom['title'].apply(keep_nouns_adjectives)
df_jomr['title'] = df_jomr['title'].apply(keep_nouns_adjectives)
df_jcr['title'] = df_jcr['title'].apply(keep_nouns_adjectives)
df_jcp['title'] = df_jcp['title'].apply(keep_nouns_adjectives)
df_jam['title'] = df_jam['title'].apply(keep_nouns_adjectives)


 ## count summary statistics for part 4
df_journals_list_part4 = [df_jom, df_jomr,df_jcr, df_jcp, df_jam ]

all_text_body_part4 = concat_texts('body',df_journals_list_part4 )
count_n_unique_top_words(all_text_body_part4, 10)

all_text_abstract_part4 = concat_texts('abstract',df_journals_list_part4 )
count_n_unique_top_words(all_text_abstract_part4, 10)

all_text_titles_part4 = concat_texts('title',df_journals_list_part4 )
count_n_unique_top_words(all_text_titles_part4, 10)




# to gzip 
df_jom.to_parquet("../Data/clean/journalofmarketing_adject_nouns.gzip", compression='gzip')
df_jomr.to_parquet("../Data/clean/journalofmarketingresearch_adject_nouns.gzip", compression='gzip')
df_jcr.to_parquet("../Data/clean/journalofconsumerresearch_adject_nouns.gzip", compression='gzip')
df_jcp.to_parquet("../Data/clean/journalofconsumerpsych_adject_nouns.gzip", compression='gzip')
df_jam.to_parquet("../Data/clean/journalacademyofmarketingscience_adject_nouns.gzip", compression='gzip')


