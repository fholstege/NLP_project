# -*- coding: utf-8 -*-


import urllib.request
import requests
import browser_cookie3
import time
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


def check_items_from_soup_search(ReturnObj):
    
    if len(ReturnObj) == 0:
        return 'NA'
    else:
        return ReturnObj[0].text


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
    author_notes_search = soup.find_all("author-notes")
    author_notes = check_items_from_soup_search(author_notes_search)

    # get abstract
    abstract_search = soup.find_all("abstract")
    abstract = check_items_from_soup_search(abstract_search)

    # extract keywords    
    keywords_search = soup.find_all("kwd-group")
    keywords = check_items_from_soup_search(keywords_search)

    # extract acknowledgements
    acknowledge_search = soup.find_all("ack")
    acknowledge = check_items_from_soup_search(acknowledge_search)

    # ref list  - raw HTML
    ref_list = soup.find_all("ref-list")[0]

    # fn group (associate editor, declaration of conflicting interest, funding, online supplement)
    fngroup_search = soup.find_all("fn-group")
    fngroup = check_items_from_soup_search(fngroup_search)
    
    
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
    
    # column names for df to be filled
    col_df = ['title', 'DOI' , 'body', 'author_notes', 'abstract', 'keywords', 'acknowledge', 'ref_list', 'fngroup']
    
    # list of dfs with data from SAGE
    list_sage_data_per_journals = []
    
    # go through each csv
    for csv_location in list_csv_locations:
        
        # get DOI list
        df_WoS = pd.read_csv(csv_location)
        list_DOI = list(df_WoS['DOI'])
        
        # turn to urls, create df to be filled
        urls_sage = create_list_urls_sage(list_DOI)
        df_sage_data = pd.DataFrame(index = range(1,len(urls_sage)+1), columns = col_df)
        
        # go through each url and pull the data
        for index_url in range(0,len(urls_sage)):
            print("{0} / {1}" .format(index_url+1, len(urls_sage)))
            
            # extract data from url
            url = urls_sage[index_url]
            title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup = scrape_article_sage(url, cookies)
            
            # putt data in row and add to df
            row_entry = [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]
            df_sage_data.iloc[index_url] = row_entry
            
            # sleep to make sure we are not recognized as DoS attack
            time.sleep(1)
        # add the df to list
        list_sage_data_per_journals.append(df_sage_data)
    
    # return list of df
    return list_sage_data_per_journals
        
        
 
# get cookies from firefox or chrome to be able to access articles
# for chrome use .chrome() or uncomment if it works for you without cookie transfer
cj = browser_cookie3.firefox() 


df_journal_of_marketing = create_dataset_sage(['Data/Raw/test.csv'], cj)


