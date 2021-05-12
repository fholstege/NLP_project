# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:12:07 2021

@author: flori
"""



def clean_in_text_references(body_html, publisher):
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
        references = body_html.find_all('a', {'class': 'link-ref'})
    elif publisher == 'Wiley':
        references = body_html.find_all('a', {'class': 'bibLink'})
    elif publisher == 'SAGE':
        references = body_html.find_all('xref')


    for ref in references:
        ref.decompose()
    
    return body_html


