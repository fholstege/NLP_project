from remove_functions import remove_in_text_references_text, remove_stopwords_non_alpha_single_words,remove_titles_sage, remove_in_text_references_text, keep_nouns_adjectives
import pandas as pd
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize 

### Part 1: raw data 


# read the zipped files
df_jom = pd.read_parquet('../Data/Scraped/journalofmarketing_data_lim.gzip')
df_jomr = pd.read_parquet('../Data/Scraped/journalofmarketingresearch_data_lim.gzip')
df_jcr = pd.read_parquet('../Data/Scraped/journalofconsumerresearch_data_lim.gzip')
df_jcp = pd.read_parquet('../Data/Scraped/journalofconsumerpsych_data_lim.gzip')
df_jam = pd.read_parquet('../Data/Scraped/journalacademyofmarketingscience_data_lim.gzip')

df_jom.apply()


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



# remove stopwords and non-alpha words
df_jom['body'] = df_jom['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jomr['body'] = df_jomr['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcr['body'] = df_jcr['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jcp['body'] = df_jcp['body'].apply(remove_stopwords_non_alpha_single_words, stopword_list = stopwords_final)
df_jam['body'] = df_jam['body'].apply(remove_stopwords_non_alpha_single_words,stopword_list = stopwords_final)



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



# to gzip 
df_jom.to_parquet("../Data/clean/journalofmarketing_words.gzip", compression='gzip')
df_jomr.to_parquet("../Data/clean/journalofmarketingresearch_words.gzip", compression='gzip')
df_jcr.to_parquet("../Data/clean/journalofconsumerresearch_words.gzip", compression='gzip')
df_jcp.to_parquet("../Data/clean/journalofconsumerpsych_words.gzip", compression='gzip')
df_jam.to_parquet("../Data/clean/journalacademyofmarketingscience_words.gzip", compression='gzip')


