

def remove_stopwords(stopword_list, word_tokens):
    
    filtered_words = [w for w in word_tokens if not w in stopword_list] 
    
    return filtered_words


def remove_non_alpha(word_tokens):
    
    alpha_words = [word for word in word_tokens if word.isalpha()]
    
    return alpha_words

def remove_single_words(word_tokens):
    
    not_single_words = [word for word in word_tokens if len(word) != 1]
    
    return not_single_words

