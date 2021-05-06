# -*- coding: utf-8 -*-


import urllib.request
import requests
import browser_cookie3
import time
from bs4 import BeautifulSoup as bs
import pandas as pd

def create_list_urls_OUP(list_DOI):
    """
    

    Parameters
    ----------
    list_DOI : list
        Takes list of the DOI's for a set of articles that needs to be scraped from the OUP website.

    Returns
    -------
    list_urls_sage : TYPE
        list of urls from the OUPz website to be scraped.

    """
    
    base_url = 'https://doi-org.eur.idm.oclc.org/'
    
    list_urls_OUP = [base_url + DOI for DOI in list_DOI]

    return list_urls_OUP



def check_items_from_soup_search(ReturnObj):
    
    if len(ReturnObj) == 0:
        return 'NA'
    else:
        return ReturnObj[0].text


def scrape_article_OUP(url, cookies): 
    """
    

    Parameters
    ----------
    url : string
        url on the OUP website to be scraped (HTML).
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
        acknowledgements in the article.
    ref_list : string
        reference list. - cleaned for HTML tags but not separated by article
    fngroup : string
        associate editor, declaration of conflicting interest, funding, online supplement.

    """

    try:
        response = requests.get(url, cookies=cookies, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()

    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return "NA"
    
    # parse xml content
    soup = bs(response.content, "html.parser")
    
    # information to crosscheck with scopus data to confirm we have correct data
    # use title 
    title = soup.find("h1", {"class": "wi-article-title article-title-main"}).text
    # or use doi
    doi = "NA"
    
    # get body of text, includes in the end data collection info and acknowledgements
    body = soup.find_all("p", {"class": "chapter-para"})
    # concatenate body of text
    if len(body) > 1:
        body = ' '.join([section.text for section in body])
    elif len(body) == 1:
        body = body[0].text
    else:
        body = "NA"
        
    # extract author notes
    author_notes = "NA"

    # get abstract
    abstract_search = soup.find_all("section", {"class": "abstract"})
    abstract = check_items_from_soup_search(abstract_search)

    # extract keywords, not easily possible as it is a java widget
    # can instead use keywords given by scopus  
    keywords = soup.find("div", {"class": "kwd-group"}).text

    # extract acknowledgements
    acknowledge = "NA"

    # ref list  - raw HTML
    ref_list = soup.find("div", {"class": "ref-list js-splitview-ref-list"})

    # fn group (associate editor, declaration of conflicting interest, funding, online supplement)
    # not easily available would require more work with little benefit?
    # can take editor from scopus data
    fngroup = "NA"
    
    return title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup
    


def create_dataset_OUP(csv_location, cookies):
    """
    

    Parameters
    ----------
    csv_locations : list
        Location of the csv file with data from Web of science.
    cookies : cookie object
        cookies for the browser.

    Returns
    -------
    df_wiley_data: dataframe
        each item is a dataframe with pulled data from SAGE.

    """
    
    # column names for df to be filled
    col_df = ['title', 'DOI' , 'body', 'author_notes', 'abstract', 'keywords', 'acknowledge', 'ref_list', 'fngroup']
    
    # save dict with stats on what was missed
    dict_missed = {'total_missed': 0,
                   'indexes_missed': []}
    
    
    # get DOI list
    df_WoS = pd.read_csv(csv_location)
    list_DOI = list(df_WoS['DOI'])
    
    # turn to urls, create df to be filled
    urls_OUP = create_list_urls_OUP(list_DOI)
    df_OUP_data = pd.DataFrame(index = range(1,len(urls_OUP)+1), columns = col_df)
    
    # go through each url and pull the data
    for index_url in range(0,len(urls_OUP)):
        print("{0} / {1}" .format(index_url+1, len(urls_OUP)))
        
        # extract data from url
        url = urls_OUP[index_url]
        
        if  scrape_article_OUP(url, cookies) == "NA":
            dict_missed['total_missed'] += 1
            dict_missed['indexes_missed'].append(index_url)
            
            continue 
        else:
            title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup = scrape_article_OUP(url, cookies)
        
        # put data in row and add to df
        row_entry = [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]
        df_OUP_data.iloc[index_url] = row_entry
        
        # sleep to make sure we are not recognized as DoS attack
        time.sleep(3)
    
    # return df
    return df_OUP_data
        
        
 
# get cookies from firefox or chrome to be able to access articles
# for chrome use .chrome() or uncomment if it works for you without cookie transfer
cj = browser_cookie3.firefox() 


df_journal_of_consumer_research = create_dataset_OUP('data/raw/journalofconsumerresearch_WoS.csv', cj)
df_journal_of_consumer_research.to_csv('data/scraped/journalofconsumerresearch_data.csv')






