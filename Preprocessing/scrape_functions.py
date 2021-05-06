# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:20:14 2021

@author: flori
"""
import urllib.request
import requests
import browser_cookie3
import time
from bs4 import BeautifulSoup as bs
import pandas as pd
from helpfunctions_preprocessing import create_list_urls, check_items_from_soup_search

def scrape_article_sage(url, cookies): 
    """
    

    Parameters
    ----------
    url : string
        url on the SAGE website to be scraped (XML).
    cookies : cookie object
        set of cookies present in current browser.

    Returns: list with the following
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
        return "NA"
    
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
    
    
    return [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]
    



def scrape_article_springer(url, cookies):
    
    
    try:
        response = requests.get(url, cookies=cookies)
        response.raise_for_status()
    except requests.HTTPError as exception:
        print(exception) # or save to logfile together with url
        return "NA"
    
    # parse xml content
    soup = bs(response.content, "html.parser")
    
    # information to crosscheck with scopus data to confirm we have correct data
    # use title 
    title = soup.find_all("h1", {"class": "c-article-title"})[0].text
    
    doi = soup.find_all('a', {'data-track-action': 'view doi'})[0].get('href')
    
    # get the abstract   
    abstract_search = soup.find_all("div", {"id": "Abs1-content"})
    abstract = check_items_from_soup_search(abstract_search)
    
    # get the acknowledgement
    acknowledge_search = soup.find_all('div', {'id': 'Ack1-content'})
    acknowledge = check_items_from_soup_search(acknowledge_search)
    
    author_notes_search = soup.find_all('div', {'id': 'author-information-content'})
    author_notes = check_items_from_soup_search(author_notes_search)
    
    # Body of text
    list_body_text = []
    sections = soup.find_all("div", {"class": "c-article-section"})
    
    for section in sections:
        # get section id
        id_section = section.get("id")
        
        # skip over abstract, stop when at notes or bibliography
        if id_section == 'Abs1-section':
            continue 
        elif id_section == 'notes-section' or id_section == 'Bib1-section':
            break
        else:         
           
            section_text = section.text
            
            list_body_text.append(section_text)
    

    
    body = ' '.join(map(str, list_body_text))
    
    ref_list_search = soup.find_all('section', {'data-title': 'References'})
    ref_list = check_items_from_soup_search(ref_list_search)
    
    list_keywords = soup.find_all('li', {'class': 'c-article-subject-list__subject'})
    keywords = ', '.join(map(str, list_body_text))
    
    fngroup_search =  soup.find_all('div', {'id': 'additional-information-content'})
    fngroup = check_items_from_soup_search(fngroup_search)
    
    return [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]
  


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
    
    return [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]



def scrape_article_wiley(url, cookies): 
    """
    

    Parameters
    ----------
    url : string
        url on the Wiley website to be scraped (HTML).
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
    title = soup.find("h1", {"class": "citation__title"}).text
    # or use doi
    doi = soup.find("a", {"class": "epub-doi"}).text
    
    # get body of text
    body = soup.find_all("section", {"class": "article-section__content"})
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
    abstract_search = soup.find_all("div", {"class": "article-section__content en main"})
    abstract = check_items_from_soup_search(abstract_search)

    # extract keywords, not easily possible as it is a java widget
    # can instead use keywords given by scopus  
    keywords = "NA"

    # extract acknowledgements
    acknowledge_search = soup.find_all("div", {"class": "header-note-content"})
    acknowledge = check_items_from_soup_search(acknowledge_search)

    # ref list  - cleaned for HTML tags but not separated by reference
    ref_list = soup.find_all("ul", {"class": "rlist separator"})[0].text

    # fn group (associate editor, declaration of conflicting interest, funding, online supplement)
    # not easily available would require more work with little benefit?
    # can take editor from scopus data
    fngroup = "NA"
    
    return [title, doi, body, author_notes, abstract, keywords, acknowledge, ref_list, fngroup]
        