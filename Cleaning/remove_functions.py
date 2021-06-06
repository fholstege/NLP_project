import re
import unicodedata
from bs4 import BeautifulSoup as bs
from nltk.tokenize import word_tokenize 
import nltk

def html_cleaner_jam(document):
    """
    

    Input
    ----------
    raw document : string
        Article form the Journal of the Academy of Marketing Science to clean. This corresponds to the raw, scraped data.

    Returns
    -------
    cleaned document : string
        Document cleaned for html-related things, e.g. tables, figures and titles. References are not removed using html, to not delete in-text references which convey meaning.

    """
    if document is None:
        return 'NA'
    else:
        # replace unicode characters
        doc = unicodedata.normalize("NFKD", document)

        # parse xml
        body_html = bs(doc, 'html.parser')

        # remove formulas
        math_tags = body_html.find_all('div', {'class': 'c-article-equation__content'})
        math2_tags = body_html.find_all('span', {'class': 'mathjax-tex'})

        # remove figures
        figure_tags = body_html.find_all('figure')

        # remove  headings
        heading_tags = body_html.find_all(["h1", "h2", "h3", "h4"])

        # make list of stuff to remove
        remove_list = heading_tags + figure_tags + math_tags + math2_tags

        # remove loop
        for s in remove_list:
            s.extract()
            
        # get text from html
        doc = body_html.get_text(separator=' ')
        
        return doc


def html_cleaner_jcp(document):
    """
    

    Input
    ----------
    raw document : string
        Article form the Journal of Consumer Psychology to clean. This corresponds to the raw, scraped data.

    Returns
    -------
    cleaned document : string
        Document cleaned for html-related things, e.g. tables, figures and titles. References are not removed using html, to not delete in-text references which convey meaning.

    """
    if document is None:
        return 'NA'
    else:
        # replace unicode characters
        doc = unicodedata.normalize("NFKD", document)

        # parse html
        body_html = bs(doc, 'html.parser')

        # remove formulas
        # math_tags = body_html.find_all('math')

        # remove figures
        figure_tags = body_html.find_all('section', {'class': 'article-section__inline-figure'})

        # remove tables
        table_tags = body_html.find_all('table')
        table2_tags = body_html.find_all('div', {'class': 'article-table-content'})

        # remove  headings
        heading_tags = body_html.find_all(["h1", "h2", "h3", "h4"])

        # make list of stuff to remove
        remove_list = heading_tags + figure_tags + table_tags + table2_tags

        # remove loop
        for s in remove_list:
            s.extract()
            
        # get text from html
        doc = body_html.get_text(separator=' ')
        
        return doc


def html_cleaner_jcr(document):
    """
    

    Input
    ----------
    raw document : string
        Article form the Journal of Consumer Research to clean. This corresponds to the raw, scraped data.

    Returns
    -------
    cleaned document : string
        Document cleaned for html-related things, e.g. tables, figures and titles. References are not removed using html, to not delete in-text references which convey meaning.

    """
    if document is None:
        return 'NA'
    else:
        # replace unicode characters
        doc = unicodedata.normalize("NFKD", document)

        # parse html
        body_html = bs(doc, 'html.parser')

        # remove page numbers
        sup_tags = body_html.find_all('sup')

        remove_list = sup_tags

        # remove loop
        for s in remove_list:
            s.extract()
            
        # get text from html
        doc = body_html.get_text(separator=' ')

        # take out headings
        doc = re.sub(r'[A-Z]{2,}[^a-z]+[A-Z]{2,}[\s\.]', '', doc)
        
        return doc

def xml_cleaner_jom_jomr(document):
    """
    

    Input
    ----------
    raw document : string
        Article form the Journal of Marketing or Journal of Marketing Research to clean. This corresponds to the raw, scraped data.

    Returns
    -------
    cleaned document : string
        Document cleaned for xml-related things, e.g. tables, figures and titles. References are not removed using xml, to not delete in-text references which convey meaning.

    """
    if document is None:
        return 'NA'
    else:
        # replace unicode characters
        doc = unicodedata.normalize("NFKD", document)
        
        # replace bold stuff to make sure words remain intact
        doc = re.sub(r'<\/?bold>', '', doc)

        # parse xml
        body_html_sage = bs(doc, 'xml')

        # remove titles of sections
        titles = body_html_sage.find_all('title')
        
        # remove headings
        heading_tags = body_html_sage.find_all(["h1", "h2", "h3", "h4"])

        # remove page numbers
        sup_tags = body_html_sage.find_all('sup')

        # remove URLs
        url_tags = body_html_sage.find_all('ext-link')

        # remove captions and labels (figures, tables, etc.)
        captions = body_html_sage.find_all('caption')
        labels = body_html_sage.find_all('label')

        # remove tables in-text
        in_text_tables = body_html_sage.find_all("table-wrap")
        # in_text_tables_footer = body_html_sage.find_all("alternatives")

        # remove in-line formula
        in_text_formula = body_html_sage.find_all('inline-formula')
        display_formula = body_html_sage.find_all('disp-formula')

        # make list of stuff to remove
        remove_list = titles + sup_tags + url_tags + captions + labels + in_text_tables + in_text_formula\
            + display_formula + heading_tags

        # remove loop
        for s in remove_list:
            s.extract()
            
        # get text from html
        doc = body_html_sage.get_text(separator=' ')
        
        return doc



# iteratively removes brackets and corresponding contents (test (test) test)
def remove_text_between_parens(text):
    n = 1  # run at least once
    while n:
        text, n = re.subn(r'\([^()]*\)', ' ', text)  # remove non-nested/flat balanced parts
    return text


def standard_cleaner(document):
    """
    

    Input
    ----------
    document : string
        Document to clean. This cleaning can be applied to all journals and all types.

    Returns
    -------
    cleaned document : string
        Document cleaned for various things. 

    """
    # remove any parentheses and stuff within
    doc = remove_text_between_parens(document)
    
    # remove greek letters
    doc = re.sub(r'[α-ωΑ-Ω]', '', doc)
    
    # change .5 to 0.5 etc.
    doc = re.sub(r'(?=\.\d+)', '0', doc)
    
    # remove \xad
    doc = re.sub(r'\xad', '', doc)

    # remove decimal points and spaces in numbers
    doc = re.sub(r'(?<=\d)[,\.\s\-](?=\d)', '' , doc)

    # replace numbers by tokens
    doc = re.sub(r' \d+\s?%?(?=[\s\.])', ' [NUM] ' , doc)
    doc = re.sub(r'[<>]?\d+(?=[\s%,;:–\.])', ' [NUM] ', doc)
    
    # remove line breaks
    doc = re.sub(r'\n', " ", doc)
    
    # remove stuff like "A third possibility—our proposal—is that" enclosed by —
    doc = re.sub(r'—[\w\d\s]+—', ' ', doc)

    # correct t -tests to t-tests
    doc = re.sub(r' t\s-test', ' t-test', doc)

    # remove dot from et al.
    doc = re.sub(r'(?<=et al)\.(?=.)', '', doc)

    # remove semicolons and other special characters as well as commas
    doc = re.sub(r'[#@;,"]', ' ', doc)

    # replace :, ?, ! by ., also several consecutive occurrences
    doc = re.sub(r'[:\?!]+', ".", doc)

    # replace U.S.A. and U.S. by US
    doc = re.sub(r'U\.S\.', 'US', doc)
    doc = re.sub(r'U\.S\.A\.', 'US', doc)

    # remove quotation marks for quotes
    doc = re.sub(r'‘(.*?)’', r'\1', doc)
    doc = re.sub(r'“(.*?)”', r'\1', doc)
    doc = re.sub(r'"(.*?)"', r'\1', doc)
    
    # remove leading - or —
    doc= re.sub(r'(?<=\s)[\-\—](?=[^\s])', '', doc)
    
    # remove free standing - or — o or minus sign or those at sentence end
    doc= re.sub(r'(?<=\s)[\-\—\−](?=[\s\.])', '', doc)

    # replace multiple dots by space
    doc = re.sub(r'\.{2,}', ' ', doc)

    # remove non-alphanumeric characters (not for BERT)
    doc = re.sub(r'[\^&*_#@&\+\[\]\|=±\/]', ' ', doc)
    
    # remove additional math symbols
    doc = re.sub(r'[><∣]', ' ', doc)

    # substitute for 'times' for multiplication
    doc = re.sub(r'×', 'times', doc)

    # ensure vs is coded correctly
    doc = re.sub(r'vs\.', 'vs', doc)
    
    # ensure etc. is etc only in sentences (no sentence ends)
    doc = re.sub(r'etc\.(?=\s[a-z])', 'etc', doc)
    
    # remove dot from Eq. and Fig.
    doc = re.sub(r'(?<=Fig)\.(?=[\s])', ' ', doc)
    doc = re.sub(r'(?<=Eq)\.(?=[\s])', ' ', doc)

    # remove double spaces
    doc = re.sub(r'\s{2,}', " ", doc)

    # remove spaces in front of dots or commas
    doc = re.sub(r'(?<=.)\s(?=[\.\,])', "", doc)

    # replace remaining (resulting from previous removing of spaces) multiple dots by single dot
    doc = re.sub(r'\.{2,}', '.', doc)
    
    # remove trailing spaces
    doc = doc.strip()

    return doc

# function to remove NUM tags
def remove_NUM_tag(document):
    return re.sub('\[NUM\]', '', document)

def remove_stopwords(stopword_list, word_tokens):
    
    filtered_words = [w for w in word_tokens if not w in stopword_list] 
    
    return filtered_words


def remove_non_alpha(word_tokens):
    
    alpha_words = [word for word in word_tokens if word.isalpha()]
    
    return alpha_words

def remove_single_words(word_tokens):
    
    not_single_words = [word for word in word_tokens if len(word) != 1]
    
    return not_single_words

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

def keep_nouns_adjectives(body_str):
    word_tokens = word_tokenize(body_str)
    tags = nltk.pos_tag(word_tokens)
    adj_nouns = [word for word,pos in tags if (pos == 'NN' or pos == 'ADJ')]
    
    adj_nouns_str = " ".join(adj_nouns)

    return adj_nouns_str
