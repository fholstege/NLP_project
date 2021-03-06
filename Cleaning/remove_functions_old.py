from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize 
import nltk
from nltk.util import ngrams
from collections import Counter





def keep_nouns_adjectives(body_str):
    word_tokens = word_tokenize(body_str)
    tags = nltk.pos_tag(word_tokens)
    adj_nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'ADJ')]
    
    adj_nouns_str = " ".join(adj_nouns)

    return adj_nouns_str


def remove_stopwords(stopword_list, word_tokens):
    
    filtered_words = [w for w in word_tokens if not w in stopword_list] 
    
    return filtered_words


def remove_non_alpha(word_tokens):
    
    alpha_words = [word for word in word_tokens if word.isalpha()]
    
    return alpha_words

def remove_single_words(word_tokens):
    
    not_single_words = [word for word in word_tokens if len(word) != 1]
    
    return not_single_words




def remove_titles_sage(body_str_sage):
    
    if body_str_sage is None:
        return 'NA'
    else:
        body_html_sage = bs(body_str_sage, 'xml')
        titles = body_html_sage.find_all('title')
        
        for title in titles:
            title.decompose()
            
        return str(body_html_sage)



def remove_in_text_references_html(body_html, publisher):
    """
    

    Parameters
    ----------
    body_html : soup object
        body of text, soup object created by html.parser (so we can apply find_all etc.).
    publisher : str
        which of the four publishers.

    Returns
    -------
    body_html : returns html without the references
        In some cases, author names remain.

    """
    
    if publisher == 'springer':
        references = body_html.find_all('a', {'data-test': 'citation-ref'})
    elif publisher == 'OUP':
        references = body_html.find_all('span', {'class': 'xrefLink'})
    elif publisher == 'Wiley':
        references = body_html.find_all('a', {'class': 'bibLink'})
    elif publisher == 'SAGE':
        references = body_html.find_all('xref')


    for ref in references:
        ref.decompose()
    
    return body_html


def remove_in_text_references_text(body_str, publisher):
    
    if publisher == 'SAGE':
        type_series = 'xml'
    else:
        type_series = 'html.parser'
    
    
    if body_str is None:
        return 'NA'
    else:
    
        body_html = bs(body_str, type_series)
        
        body_html_in_text_ref_removed = remove_in_text_references_html(body_html, publisher)
        
        return body_html_in_text_ref_removed.text



def remove_stopwords_non_alpha_single_words(body_str, stopword_list = None, alpha = True, single_words = True, stopwords = True):
    
    if type(body_str) != str:
        return 'NA'
    else:
        word_tokens = word_tokenize(body_str)
        
        if stopwords:
            word_tokens = remove_stopwords(stopword_list, word_tokens)
        
        if alpha:
            word_tokens = remove_non_alpha(word_tokens)
            
        if single_words:
            word_tokens = remove_single_words(word_tokens)
        
    
        cleaned_str = " ".join(word_tokens)
        
        return cleaned_str
    

    
