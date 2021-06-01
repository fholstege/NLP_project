
import random
from gensim.models import ldamodel
import logging
from gensim.models.coherencemodel import CoherenceModel
import sklearn.metrics as metrics
import numpy as np

def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))



def get_K_folds_from_corpus_list(corpus_list, K):
    """
    

    Parameters
    ----------
    corpus_list : list
        list with corpus of text.
    K : int
        number of folds.

    Returns
    -------
    overview_indeces_fold : list of lists 
        each element is a list with indeces of fold.

    """
    
    
    size = len(corpus_list)
    size_fold = int(round(size/K,0))
    indeces_corpus = [i for i in range(0, len(corpus_list))]
    

    overview_indeces_fold = []
    
    for i in range(0,K):
        
        if size_fold > len(indeces_corpus):
            size_fold = len(indeces_corpus)
       
        # pick indeces for fold
        indeces_fold = random.sample(indeces_corpus, size_fold)
        
        indeces_corpus = [index for index in indeces_corpus if index not in indeces_fold] 
        
        if i == K-1:
            
            if len(indeces_corpus) != 0:
                
                indeces_fold += indeces_corpus
        
        overview_indeces_fold.append(indeces_fold)
    
    return overview_indeces_fold


            

def get_perplexity_coherence_from_LDA_CV(corpus_list, n_topics, id2word, K, coherence_measure='u_mass'):
    """
    

    Parameters
    ----------
    corpus_list : list
        list with corpus of text.
    n_topics : int
        number of topics.
    id2word : -
        dictionary for lda.
    K : int
        number of folds.

    Returns
    -------
    average log perplexity across 

    """
    
    list_of_fold_indeces = get_K_folds_from_corpus_list(corpus_list, K)
    fold_perplexity = []
    fold_coherence = []
    
    for fold_indeces in list_of_fold_indeces:
        
        fold = [corpus_list[i] for i in fold_indeces]
        training_corpus = [corpus_list[j] for j in range(0, len(corpus_list)) if j not in fold_indeces]
    
        lda_model = ldamodel.LdaModel(corpus=training_corpus, id2word=id2word, num_topics=n_topics)
        
        log_perplexity_fold = lda_model.log_perplexity(fold)
        coherence_model = CoherenceModel(model = lda_model, corpus=fold, coherence=coherence_measure)
        
        fold_perplexity.append(log_perplexity_fold)
        fold_coherence.append(coherence_model.get_coherence())
    
    avg_perplexity = sum(fold_perplexity) / len(fold_perplexity) 
    avg_coherence = sum(fold_coherence)/ len(fold_coherence)
    
    return avg_perplexity, avg_coherence

def check_n_topic_scores_CV(corpus_list, range_topics, id2word, K, coherence_measure = 'u_mass'):
    
        
    # implement cross validation
    result_per_n_topics = []
    
    for n_topics in range_topics:
        
        avg_perplexity, avg_coherence = get_perplexity_coherence_from_LDA_CV(corpus_list, n_topics, id2word=id2word, K=K, coherence_measure = coherence_measure)
        result_dict = {'avg_perplexity': avg_perplexity, 'avg_coherence':avg_coherence }
        
        result_per_n_topics.append(result_dict)
    
    
    dict_results = dict(enumerate(result_per_n_topics, start = 2))
    return dict_results

    
