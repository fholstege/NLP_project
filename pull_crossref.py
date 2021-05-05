# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:42:37 2021

@author: flori
"""

import urllib.request
import requests
import browser_cookie3
from bs4 import BeautifulSoup as bs

# FUNCTIONS

def scrape_article(url, cookies): 
    '''
    extracts text and meta data of journal article from DOI
    
    input:
        url to article - string
        add_cookies - boolean, if want to use cookies
        cookies - browser cookies to be able to download restricted articles
    output:
        scraped meta info and text - dataframe
    '''
    # request article
    
    try:
        
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()

    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return
    
    # parse xml content
    soup = bs(response.content, "xml")
    print(soup)
    
    # information to crosscheck with scopus data to confirm we have correct data
    # use title 
    title = soup.find_all("article-title")[0]
    # or use doi
    doi = soup.find_all("article-id")[0]
    
    # get body of text
    body = soup.body    

    # extract author notes
    author_notes = soup.find_all("author-notes")

    # get abstract
    abstract = soup.find_all("abstract")

    # extract keywords
    keywords = soup.find_all("kwd-group")

    # extract acknowledgements
    acknowledge = soup.find_all("ack")

    # ref list 
    ref_list = soup.find_all("ref-list")

    # fn group (associate editor, declaration of conflicting interest, funding, online supplement)
    fngroup = soup.find_all("fn-group")
    
    return title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup
    # eventually we could just add a row to the dataframe and save output directly
    
 
# get cookies from firefox or chrome to be able to access articles
# for chrome use .chrome() or uncomment if it works for you without cookie transfer
cj = browser_cookie3.firefox() 
cj

url = 'https://journals-sagepub-com.eur.idm.oclc.org/doi/full-xml/10.1177/00222429211003690'
url_2 = 'https://journals-sagepub-com.eur.idm.oclc.org/doi/full-xml/10.1177/0022242920985784'

title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup = scrape_article(url_2, cj)


# NOTES
# extract all references to be able to build a network? or just get total number of references from crossref?

# scrape fields of referenced papers? e.g. to see if marketing references other fields? not sure how
# to get the field though? maybe from the journal name of the referenced paper? check them against scopus or sth?