

# Packes required for subsequent analysis. P_load ensures these will be installed and loaded. 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(reticulate,
               tidyverse,
               glmnet,
               pscl,
               rlist,
               parallel,
               gbm, 
               devtools,
               reshape2,
               stargazer,
               gridExtra,
               caTools,
               caret,
               Metrics) 


# add repository to make mclapply() run in parallel (only necessary on windows)
install_github('nathanvan/parallelsugar')
library(parallelsugar)

# create ability to read parquet files into r
pandas <- import("pandas")

#################
# read_parquet: reads parquet (.gzip) files
# 
#   Arguments: 
#       path: location of the parquet file
# 
#   return:
#       df: data.frame with file
#################
read_parquet <- function(path, columns = NULL) {
  
  path <- path.expand(path)
  path <- normalizePath(path)
  
  if (!is.null(columns)) columns = as.list(columns)
  
  xdf <- pandas$read_parquet(path, columns = columns)
  
  xdf <- as.data.frame(xdf, stringsAsFactors = FALSE)
  
  dplyr::tbl_df(xdf)
  
}
getwd()
setwd('C:/Users/flori/OneDrive/Documents/GitHub/NLP_project/Models')

# read in full dataset, merge
df_full = read_parquet('../data/clean/all_journals_BERT_merged.gzip')
df_BERT <- read_parquet('../data/representations/BERT_all_embeddings.gzip')
df_SCIBERT <- read_parquet('../data/representations/BERT_all_embeddings.gzip')
df_ROBERTA <- read_parquet('../data/representations/BERT_all_embeddings.gzip')
df_BERT$year <- df_full$year
df_SCIBERT$year <- df_full$year
df_ROBERTA$year <- df_full$year


vec_poolings =colnames(df_BERT)[2:31]

df_BERT_final <- df_BERT %>%
  select(citations, year, vec_poolings)

df_SCIBERT_final <- df_SCIBERT %>%
  select(citations, year, vec_poolings)

df_ROBERTA_final <- df_ROBERTA %>%
  select(citations, year, vec_poolings)



##############################
# Helpfunctions 
##############################



################################
# 2) create df's for prediction
################################


####################
# create_df_for_prediction; taking one of the df's with all embeddings, extracts for a particular type of pooling a dataframe that can be used for prediction 
####################

create_df_for_prediction <- function(name_representation, df){
  
  df_list_representation = df[name_representation]
  n_dimensions <-length( unlist(df_list_representation[1,]))
  
  df_representation  <-  as.data.frame(matrix(unlist(df_list_representation), 
                                              ncol=n_dimensions))
  
  colnames(df_representation) <- paste0(name_representation, '_', seq(1, n_dimensions) )
  
  df_representation$citations <-df$citations
  df_representation$year <-df$year
  
  
  return(df_representation)
  
}

# function to speed up cv.glmnet in lapply with different alpha's 
cv_glmnet_wrapper <- function(alpha, x_vars, y_var, lambda_cv, K,type_measure='mse', ...){
  
  
  cv_result <- cv.glmnet(x_vars, 
                         y_var,
                         alpha = alpha, 
                         lambda = lambda_cv, 
                         nfolds = K,
                         type.measure = type_measure,
                         ...)
  
  return(cv_result)
}

# function to get results from the wrapper function
get_results_cv_wrapper <- function(x){ 
  score = min(x$cvm)
  lambda = x$lambda.min
  
  return(c(score, lambda))
}

# ensures we can apply gbm in lapply
gbm_inGridsearch <- function(l_param, obj_formula, distribution, df_var, n_trees, K){

  
  result <- gbm(formula = obj_formula, distribution = distribution, data = df_var,  n.trees = n_trees, interaction.depth = l_param$interaction_depth, n.minobsinnode = l_param$min_obs,
                cv.folds = K, 
                n.cores = 4,
               )
  
  return(result)
}





# creates df to analyse tree results
create_df_TreeResult <- function(obj_tree_result){
  
  n_tree_best =  which.min(obj_tree_result$cv.error)
  best_cv_error = obj_tree_result$cv.error[n_tree_best]
  
  dfTreeResult <- data.frame(n_tree =n_tree_best , cv_error = best_cv_error, min_obs = obj_tree_result$n.minobsinnode, interaction_depth = obj_tree_result$interaction.depth)
  return(dfTreeResult)
}


# param grid search for gbm function
grid_search_gbm <- function(obj_formula, df_var,df_param,K, n_trees){
  
  # create list with all parameters
  l_params =  split(df_param, seq(nrow(df_param)))
  

  
  # apply gbm to all combinations 
  l_result_grid <- lapply(l_params ,gbm_inGridsearch,obj_formula = obj_formula, distribution = "gaussian", df_var = df_var, n_trees = n_trees, K=K)#, mc.cores = detectCores()-2)
  
  # save results of the data.frame
  l_result_grid_clean <- lapply(l_result_grid, create_df_TreeResult)
  
  df_results <- do.call("rbind", l_result_grid_clean)
  
  return(df_results)
}





####################
# get_predictions: obtains predictions on a test sample, based on a model object
####################


get_predictions <- function(df_predictions, 
                            split_ratio = 0.7,
                            log_transformation = FALSE,
                            modelObj=NA,
                            cv_param_choose = FALSE,
                            type_model = NA,
                            alpha_cv = NA, 
                            lambda_cv = NA,
                            df_param_gbm = NA,
                            n_trees = NA,
                            K=NA,...){
  
  train_sample <- sample.split(df_predictions$citations, SplitRatio = split_ratio)
  df_train <- subset(df_predictions, train_sample == TRUE)
  df_test <- subset(df_predictions, train_sample == FALSE)
  
  
  
  ####################
  # Because the elastic net uses a very different data structure, we need to use this version
  ####################
  if(type_model == 'elastic net'){
    
    print("CORRECT")
    
    model_matrix_train <- model.matrix(citations~. , df_train)
    model_matrix_test <- model.matrix(citations~. , df_test)
    
    x_var_train <- model_matrix_train[,-1]
    x_var_test <-model_matrix_test[,-1]
    y_var_test <- df_test$citations

    if(log_transformation){
      y_var_train_used <- log(1 + df_train$citations)
    }else{
      y_var_train_used <-  df_train$citations
    }
    
    if(cv_param_choose){
      list_cv_results <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_var_train), y = y_var_train_used, lambda_cv = lambda_cv, K=K)
      df_result_cv <- data.frame(list.rbind(lapply(list_cv_results, get_results_cv_wrapper)), alpha = alpha_cv)
      colnames(df_result_cv)[1:2] <- c('MSE', 'Lambda')
      
      opt_index <- which.min(df_result_cv$MSE)
      lambda_cv <- df_result_cv[opt_index,]$Lambda
      alpha_cv <- df_result_cv[opt_index,]$alpha
      
    }
   
    
    elastic_net_model <- glmnet(as.matrix(x_var_train), y_var_train_used, lambda = lambda_cv, alpha = alpha_cv)
    predictions_elastic_net <- predict(elastic_net_model, as.matrix(x_var_test))
    
    if(log_transformation){
      predictions_elastic_net <- exp(predictions_elastic_net) - 1
    }
    
    MSE <- mean((predictions_elastic_net - y_var_test)^2)
    MAE <- mean(abs(predictions_elastic_net - y_var_test))
    
    result = list('predictions'= c(predictions_elastic_net), 'MSE' = MSE, 'MAE' = MAE)

  } else if(type_model == 'gbm'){

    if(log_transformation){
      
      result_grid_search <- grid_search_gbm(log(1 + citations) ~ . , df_var =df_train, df_param_gbm , K, n_trees)
    }else{
      result_grid_search <- grid_search_gbm( citations ~ . , df_var =df_train, df_param_gbm , K, n_trees)
    }
    
    
    print('grid search done')
    
    opt_index <- which.min(result_grid_search$cv_error)
    opt_n_tree <- result_grid_search[opt_index, ]$n_tree
    opt_min_obs <- result_grid_search[opt_index, ]$min_obs
    opt_interaction_depth <- result_grid_search[opt_index, ]$interaction_depth
    
    boosted_model <- gbm(citations ~ . , data =df_train, distribution = 'gaussian', n.trees = opt_n_tree, interaction.depth = opt_interaction_depth, n.minobsinnode = opt_min_obs)
    
    predictions <- predict(boosted_model, df_test)
    
    if(log_transformation){
      predictions <- exp(predictions) - 1
    }
    
    MSE <- mean((predictions - df_test$citations)^2)
    MSE <- mean(abs(predictions - df_test$citations))
    
    
    result = list('predictions'= c(predictions), 'MSE' = MSE,'MAE' = MAE)
    
  }else{ #for all other models
  
  
    if(log_transformation){
      
      model_train <- modelObj(log( 1 + citations) ~ ., data = df_train, ...)
      predictions <- predict(model_train, df_test)
      predictions_backtransformed <- exp(predictions) - 1
      
      MSE <- mean((predictions_backtransformed - df_test$citations)^2)
      MAE <- mean(abs(predictions_backtransformed - df_test$citations))
      
      result = list('predictions'= predictions_backtransformed,'MSE' = MSE ,'MAE' = MAE)
      
      
    }else{
      print('incorrect')
      
      
      model_train <- modelObj(citations ~ ., data = df_train, ...)
      predictions <- predict(model_train, df_test)
      
      MSE <- mean((predictions - df_test$citations)^2)
      MAE <- mean(abs(predictions - df_test$citations))
      
      result = list('predictions'= predictions,'MSE' = MSE, 'MAE' = MAE)
      
    }
  }
  return(result)
  
  
}


####################
# compare_BERT_predictions: for particular BERT and model, get every pooling type and then check results
####################



compare_BERT_predictions <- function(df, vector_poolings, modelObj, type_model = NA, cv_param_choose = FALSE,log_transformation = FALSE,lambda_cv = NA, alpha_cv = NA,K=NA,n_trees = 200,...){
  
  
  list_result_per_pooling <- lapply(vec_poolings, function(name_pooling){
    
    df_prediction <- create_df_for_prediction(name_pooling, df)
    
    print(paste0('Calculating the results for ', name_pooling))
    
    result <- get_predictions(df_predictions = df_prediction, 
                              split_ratio = 0.7,
                              type_model = type_model,
                              log_transformation = log_transformation, 
                              modelObj = modelObj,
                              cv_param_choose = cv_param_choose, 
                              lambda_cv = lambda_cv,
                              alpha_cv = alpha_cv, 
                              K=K, 
                              n_trees = n_trees,
                              ...)
    return(result)
    
  })
  
  df_MSE_pooling <- data.frame(MSE = unlist(lapply(list_result_per_pooling, function(result){return(result$MSE)})), Pooling = vec_poolings)
  df_MAE_pooling <- data.frame(MAE = unlist(lapply(list_result_per_pooling, function(result){return(result$MAE)})), Pooling = vec_poolings)
  
  
  list_predictions <- lapply(list_result_per_pooling, function(result){return(result$predictions)})
  df_predictions_pooling = t(list.rbind(list_predictions))
  colnames(df_predictions_pooling) <- vec_poolings
  
  
  return(list("df_MSE" = df_MSE_pooling,"df_MAE" = df_MAE_pooling ,"df_predictions" = df_predictions_pooling))
  
}

###########
# Linear regression results 
###########



# define linear regression with MAE
#maeSummary <- function (data,
#                        lev = NULL,
#                        model = NULL) {
  
  
#  out <- mae(data$obs, data$pred)  
#  names(out) <- "MAE"
#  out
#}

#mControl <- trainControl(summaryFunction = maeSummary)


# get results for MAE 
results_linearRegression_BERT <- compare_BERT_predictions(df_BERT_final, vec_poolings, modelObj = lm, type_model = 'lm type',log_transformation = FALSE)
results_linearRegression_SCIBERT <- compare_BERT_predictions(df_SCIBERT_final, vec_poolings, modelObj = lm,  type_model = 'lm type',log_transformation = FALSE)
results_linearRegression_ROBERTa <- compare_BERT_predictions(df_ROBERTA_final, vec_poolings,  modelObj = lm,  type_model = 'lm type',log_transformation = FALSE)#train, method = "lm", metric = "MAE", maximize = FALSE, trControl = mControl)



###########
# Elastic Net
############

result_elasticNet_BERT <- compare_BERT_predictions(df_BERT_final, vec_poolings, modelObj = NA, type_model = 'elastic net',cv_param_choose = TRUE, alpha_cv = c(0,0.5,1), lambda_cv = c( 0.1, 1, 10,100), K = 4)
result_elasticNet_SCIBERT <- compare_BERT_predictions(df_SCIBERT_final, vec_poolings, modelObj = NA,type_model = 'elastic net', cv_param_choose = TRUE, alpha_cv = c(0,0.5,1), lambda_cv = c( 0.1, 1, 10,100), K = 4)
result_elasticNet_ROBERTA <- compare_BERT_predictions(df_ROBERTA_final, vec_poolings, modelObj = NA,type_model = 'elastic net', cv_param_choose = TRUE, alpha_cv = c(0,0.5,1), lambda_cv = c( 0.1, 1, 10,100), K = 4)





############
# GBM
############
interaction_depth_seq = c(1,3)
n_min_obs_node_seq = c(100,200)

df_param <- expand.grid(interaction_depth_seq, n_min_obs_node_seq)
colnames(df_param) <- c('interaction_depth', 'min_obs')



result_GBM_BERT <- compare_BERT_predictions(df_BERT_final, vec_poolings, modelObj = gbm,cv_param_choose = TRUE, df_param_gbm = df_param, K = 4, type_model = 'gbm', n_trees = 200)
