# -*- coding: utf-8 -*-


import urllib.request
import requests
import browser_cookie3
from bs4 import BeautifulSoup as bs
import pandas as pd

def create_list_urls_sage(list_DOI):
    """
    

    Parameters
    ----------
    list_DOI : list
        Takes list of the DOI's for a set of articles that needs to be scraped from the SAGE website.

    Returns
    -------
    list_urls_sage : TYPE
        list of urls from the SAGE website to be scraped.

    """
    
    base_url = 'https://journals-sagepub-com.eur.idm.oclc.org/doi/full-xml/'
    
    list_urls_sage = [base_url + DOI for DOI in list_DOI]

    return list_urls_sage



def scrape_article_sage(url, cookies): 
    """
    

    Parameters
    ----------
    url : string
        url on the SAGE website to be scraped (XML).
    cookies : cookie object
        set of cookies present in current browser.

    Returns
    -------
    title : string
        title of article.
    doi : string
        doi of article.
    body : string
        body of text for the article. - Raw HTML
    author_notes : string
        notes from the author.
    abstract : string
        abstract of the article.
    keywords : string
        keywords associated with the article
    acknowledge : string
        ackknowledgements in the article.
    ref_list : string
        reference list. - Raw HTML
    fngroup : string
        associate editor, declaration of conflicting interest, funding, online supplement.

    """

    try:
        
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()

    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return
    
    # parse xml content
    soup = bs(response.content, "xml")
    
    # information to crosscheck with scopus data to confirm we have correct data
    # use title 
    title = soup.find_all("article-title")[0].text
    # or use doi
    doi = soup.find_all("article-id")[0].text
    
    # get body of text
    body = soup.body    

    # extract author notes
    author_notes = soup.find_all("author-notes")
    
    if len(author_notes) == 0:
        author_notes = 'NA'
    else:
        author_notes = author_notes[0].text

    # get abstract
    abstract = soup.find_all("abstract")
    
    if len(abstract) == 0:
        abstract = 'NA'
    else:
        abstract = abstract[0].text

    # extract keywords
    keywords = soup.find_all("kwd-group")[0].text

    # extract acknowledgements
    acknowledge = soup.find_all("ack")
    
    if len(acknowledge) == 0:
        acknowledge = 'NA'
    else:
        acknowledge = acknowledge[0].text

    # ref list  - raw HTML
    ref_list = soup.find_all("ref-list")[0]

    # fn group (associate editor, declaration of conflicting interest, funding, online supplement)
    fngroup = soup.find_all("fn-group")
    
    if len(fngroup) == 0:
        fngroup = 'NA'
    else:
        fngroup = fngroup[0].text
    
    return title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup
    


def create_dataset_sage(list_csv_locations, cookies):
    """
    

    Parameters
    ----------
    list_csv_locations : list
        Location of the csv files with data from Web of science.
    cookies : cookie object
        cookies for the browser.

    Returns
    -------
    list_sage_data_per_journals : list
        each item is a dataframe with pulled data from SAGE.

    """
    
    col_df = ['title', 'DOI' , 'body', 'author_notes', 'abstract', 'keywords', 'acknowledge', 'ref_list', 'fngroup']
    
    list_sage_data_per_journals = []

    for csv_location in list_csv_locations:
        
        df_WoS = pd.read_csv(csv_location)
        
        list_DOI = list(df_WoS['DOI'])
        
        urls_sage = create_list_urls_sage(list_DOI)
        df_sage_data = pd.DataFrame(index = range(1,len(urls_sage)+1), columns = col_df)
        
        for index_url in range(0,len(urls_sage)):
            print("{0} / {1}" .format(index_url, len(urls_sage)))
            
            url = urls_sage[index_url]
            
            title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup = scrape_article_sage(url, cookies)
            
            row_entry = [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]

            df_sage_data.iloc[index_url] = row_entry
      
        list_sage_data_per_journals.append(df_sage_data)
    
    return list_sage_data_per_journals
        
        
 
# get cookies from firefox or chrome to be able to access articles
# for chrome use .chrome() or uncomment if it works for you without cookie transfer
cj = browser_cookie3.firefox() 


df_journal_of_marketing = create_dataset_sage(['Data/Raw/journalofmarketing.csv'], cj)


