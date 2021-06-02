##################################
# Author: Floris Holstege
# Goal: Predict the number of citations
#
# Parts: 
#   1) Load in packages, define help functions
#   2) Define dependent variable
#   3) check topics from LDA
#   4) Use word embeddings from Word2vec
#   5) Use tf-idf and PCA 
#   6) Use BERT embeddings
#   7) combine information
#################################



################################
# 1) Load in packages, define help functions 
################################


# Packes required for subsequent analysis. P_load ensures these will be installed and loaded. 
if (!require("pacman")) install.packages("pacman")
pacman::p_load(reticulate,
               tidyverse,
               glmnet,
               pscl,
               rlist,
               parallel,
               gbm, 
               devtools) 


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


# read in full dataset
df_full = read_parquet('../data/clean/data_journals_merged_topics.gzip')
colnames(df_full)

############################
# 2) Define dependent variable
############################


# unlogged
ggplot(data = df_full,aes(x = `Cited by`) ) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations', y = 'Frequency') + 
  theme_bw()

# logged (+1)
ggplot(data = df_full,aes(x = log(1 + `Cited by`) )) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations (Logarithm)', y = 'Frequency') + 
  theme_bw()

# Altmetric 
ggplot(data = df_full,aes(x = `Altmetric Attention Score` )) + 
  geom_histogram(bins = 20, color="black", fill="red") +
  labs(x = 'Altmetric Attention Score', y = 'Frequency') + 
  theme_bw()

# Altmetric  looged
ggplot(data = df_full,aes(x = log(1 + `Altmetric Attention Score`) )) + 
  geom_histogram(bins = 20, color="black", fill="red") +
  labs(x = 'Altmetric Attention Score', y = 'Frequency') + 
  theme_bw()


# conclusion; for prediction, keep logged. For inference, build two-stage model
# check percentage of zero citations per
n_0_log_citation = nrow(df_full[log( 1 + df_full$`Cited by`) == 0,])
n_0_log_citation/ nrow(df_full)

# conclusion; for prediction, keep logged. For inference, build two-stage model
# check percentage of zero citations per
n_0_log_altmetric = nrow(df_full[log( 1 + df_full$`Altmetric Attention Score`) == 0,])
n_0_log_altmetric/ nrow(df_full[!is.na(df_full$`Altmetric Attention Score` ),])


############################
# 3) LDA topics
############################


# not add topic 3 - since all classified in one of these
topic_var = c('Topic')
dependent_var = c('Cited by', 'Altmetric Attention Score')

# removes a couple of empty rows
df_topics <- df_full[,c(dependent_var, topic_var)]


# check for the entire dataset if the number of citations is zero
df_topics$citations_is_zero = ifelse(df_topics$`Cited by` == 0, 1, 0)


# create dataset where only citations are included that are not zero
df_model_non_zero_citations <- df_topics[df_topics$`Cited by` != 0,]

## number of citations

# run the binary, logit model for entire dataset
binary_model_topic = glm(factor(citations_is_zero) ~  factor(Topic), data = na.omit(df_topics %>% select(-`Altmetric Attention Score`)), family = binomial(link = "logit"), control = list(maxit = 50))

# linear model for number of citations >0 
linear_model_topic = lm(log(1 + `Cited by`) ~ factor(Topic)   , data = na.omit(df_model_non_zero_citations%>% select(-`Altmetric Attention Score`)))

summary(binary_model_topic)
summary(linear_model_topic)

## altmetric interest
df_topics$altmetric_is_zero = ifelse(df_topics$`Altmetric Attention Score` == 0, 1,0)
df_topics_non_zero_alt <- df_topics[df_topics$`Altmetric Attention Score` != 0,]

# run the binary, logit model for entire dataset
binary_model_topic_altmetric = glm(factor(altmetric_is_zero) ~  factor(Topic), data = na.omit(df_topics %>% select(-`Cited by`)), family = binomial(link = "logit"), control = list(maxit = 50))

# linear model for number of altmetric >0 
linear_model_topic_altmetric = lm(log(1 + `Altmetric Attention Score`) ~ factor(Topic)   , data = na.omit(df_topics_non_zero_alt%>% select(-`Cited by`)))

summary(binary_model_topic_altmetric)
summary(linear_model_topic_altmetric)




############################
# 4) Word2vec
############################


# get dataframes 
df_word2vec_mean_representation = read_parquet('../data/representations/word2vec_doc_representation_mean.gzip') %>% na.omit()
df_word2vec_max_representation = read_parquet('../data/representations/word2vec_doc_representation_max.gzip')  %>% na.omit()
df_word2vec_min_representation = read_parquet('../data/representations/word2vec_doc_representation_min.gzip')%>% na.omit()


# get x and y values
x_vars_mean <- model.matrix(`Cited by`~. , na.omit(df_word2vec_mean_representation))[,-1]
x_vars_max <-  model.matrix(`Cited by`~. , na.omit(df_word2vec_max_representation))[,-1]
x_vars_min <- model.matrix(`Cited by`~. , na.omit(df_word2vec_min_representation))[,-1]
y_var <- df_word2vec_mean_representation$`Cited by`
y_var_logged <- log(1 + y_var)


# try out these values for alpha and lambda
lambda_cv = 10^seq(2, -2, by = -0.5)
alpha_cv = seq(0,1,by=0.5)

# Splitting the data into test and train
set.seed(123)
train_indeces = which(df_word2vec_mean_representation$train_set == 1)


# mean representation
x_vars_mean_train = data.frame(x_vars_mean[train_indeces,]) %>% select(-train_set)
x_vars_mean_test = data.frame(x_vars_mean[-train_indeces,])  %>% select(-train_set)

# max representation
x_vars_max_train = data.frame(x_vars_max[train_indeces,]) %>% select(-train_set)
x_vars_max_test = data.frame(x_vars_max[-train_indeces,]) %>% select(-train_set)

# min representation
x_vars_min_train = data.frame(x_vars_min[train_indeces,]) %>% select(-train_set)
x_vars_min_test = data.frame(x_vars_min[-train_indeces,]) %>% select(-train_set)

# dependent variable
y_var_train = y_var[train_indeces]
y_var_test = y_var[-train_indeces]
y_var_train_logged = y_var_logged[train_indeces]
y_var_test_logged = y_var_logged[-train_indeces]



####
# First; baseline model with linear regression
####

linear_model_mean_rep <- lm(y_var_train_logged ~ . , data = data.frame(x_vars_mean_train))
linear_model_max_rep <- lm(y_var_train_logged ~ ., data = data.frame(x_vars_max_train))
linear_model_min_rep <- lm(y_var_train_logged ~ ., data = data.frame(x_vars_min_train))

pred_linear_model_mean_log <- predict(linear_model_mean_rep, data.frame(x_vars_mean_test))
pred_linear_model_max_log <- predict(linear_model_max_rep, data.frame(x_vars_max_test))
pred_linear_model_min_log <- predict(linear_model_min_rep, data.frame(x_vars_min_test))

pred_linear_model_mean_backtransformed <- exp(pred_linear_model_mean_log) - 1
pred_linear_model_max_backtransformed <- exp(pred_linear_model_max_log) - 1
pred_linear_model_min_backtransformed <- exp(pred_linear_model_min_log) - 1


MAE_linear_model_mean <- mean(abs(pred_linear_model_mean_backtransformed - y_var_test))
MAE_linear_model_max <- mean(abs(pred_linear_model_max_backtransformed - y_var_test))
MAE_linear_model_min <- mean(abs(pred_linear_model_min_backtransformed - y_var_test))



####
# Second; elastic net
####


# number of folds
K = 5

# function to speed up cv.glmnet in lapply with different alpha's 
cv_glmnet_wrapper <- function(alpha, x_vars, y_var, lambda_cv, K, ...){
  
  
  cv_result <- cv.glmnet(x_vars, 
            y_var,
            alpha = alpha, 
            lambda = lambda_cv, 
            nfolds = K,
            ...)
  
  return(cv_result)
}

as.matrix(x_vars_mean_train)

# cv for mean representation
cv_mean_rep <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_mean_train), y = y_var_train_logged, lambda_cv = lambda_cv, K=K)


# cv for max representation
cv_max_rep <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_max_train), y = y_var_train_logged, lambda_cv = lambda_cv, K=K)

# cv for min representation
cv_min_rep <-lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_min_train), y = y_var_train_logged, lambda_cv = lambda_cv, K=K)


# function to get results from the wrapper function
get_results_cv_wrapper <- function(x){ 
  score = min(x$cvm)
  lambda = x$lambda.min
  
  return(c(score, lambda))
  }

# get cv results for mean representation
cv_result_mean <- data.frame(list.rbind(lapply(cv_mean_rep, get_results_cv_wrapper)), alpha = alpha_cv, type.measure = 'mae')
colnames(cv_result_mean)[1:2] <- c('MAE', 'Lambda')


# get cv results for max representation
cv_result_max <- data.frame(list.rbind(lapply(cv_max_rep, get_results_cv_wrapper)), alpha = alpha_cv,  type.measure = 'mae')
colnames(cv_result_max)[1:2] <- c('MAE', 'Lambda')

# get cv results for min representation
cv_result_min <- data.frame(list.rbind(lapply(cv_min_rep, get_results_cv_wrapper)), alpha = alpha_cv,  type.measure = 'mae')
colnames(cv_result_min)[1:2] <- c('MAE', 'Lambda')


# based on results, pick lambda and alpha
opt_alpha = 0.5
opt_lambda_min = 0.01

# Rebuilding the model with best lamda value identified
lasso_model_word2vec <- glmnet(as.matrix(x_vars_min_train), y_var_train_logged, alpha = opt_alpha, lambda = opt_lambda_min)
pred_elastic_log <- predict(lasso_model_word2vec, newx = as.matrix(x_vars_min_test))

# get MAE in log
MAE_log <- mean(abs(pred_elastic_log - y_var_test))

# get the backtransformed predictions
pred_elastic_backtransformed <- exp(pred_elastic_log)-1
residuals_backtransformed <- (abs(pred_elastic_backtransformed - y_var_test))
MAE <- mean(residuals_backtransformed)

# plot residuals
colnames(residuals_backtransformed) <- 'residuals'
ggplot(data = data.frame(residuals_backtransformed), aes(x = residuals)) + 
  geom_histogram(fill = 'grey', bins = 50) + 
  labs(x = 'Residuals - MAE, Elastic Net', y = 'Frequency') + 
  theme_bw()



####
# Third; random forest 
####

df_gbm_mean_rep_train = data.frame(y_var_train_logged, x_vars_mean_train)
df_gbm_max_rep_train = data.frame(y_var_train_logged, x_vars_max_train)
df_gbm_min_rep_train = data.frame(y_var_train_logged, x_vars_min_train)


n_max_trees = 300
interaction_depth_seq = c(1,3,5)
n_min_obs_node_seq = c(100,250,500)

df_param <- expand.grid(interaction_depth_seq, n_min_obs_node_seq)
colnames(df_param) <- c('interaction_depth', 'min_obs')




# ensures we can apply gbm in lapply
gbm_inGridsearch <- function(l_param, obj_formula, distribution, df_var, K, n_max_trees){
  result <- gbm(formula = obj_formula, distribution = distribution, data = df_var, cv.folds = K, n.trees = n_max_trees, interaction.depth = l_param$interaction_depth, n.minobsinnode = l_param$min_obs)
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
grid_search_gbm <- function(obj_formula, df_var,df_param,K, n_max_trees){
  
  # create list with all parameters
  l_params =  split(df_param, seq(nrow(df_param)))
  
  # apply gbm to all combinations 
  l_result_grid <- parallelsugar::mclapply(l_params ,gbm_inGridsearch,obj_formula = obj_formula, distribution = "gaussian", df_var = df_var, K = K, n_max_trees = n_max_trees,  mc.cores = detectCores()-1)
  
  # save results of the data.frame
  l_result_grid_clean <- lapply(l_result_grid, create_df_TreeResult)
  
  df_results <- do.call("rbind", l_result_grid_clean)
  
  return(df_results)
}

df_gridsearch_result_mean <- grid_search_gbm(y_var_train_logged ~ . , df_var =df_gbm_mean_rep_train, df_param , K, n_max_trees)
df_gridsearch_result_max <- grid_search_gbm(y_var_train_logged ~ . , df_var =df_gbm_max_rep_train, df_param , K, n_max_trees)
df_gridsearch_result_min <- grid_search_gbm(y_var_train_logged ~ . , df_var =df_gbm_min_rep_train, df_param , K, n_max_trees)

boosted_model <- gbm(y_var_train_logged ~ . , data =df_gbm_min_rep, distribution = 'gaussian', n.trees = 215, interaction.depth = 3, n.minobsinnode = 100)


df_gbm_min_rep_test <-data.frame(y_var_test_logged, x_vars_min_test)
predict_gbm_logged <- predict(boosted_model, df_gbm_min_rep_test)
predict_gbm_backtransformed <- exp(predict_gbm_logged) + 1
MAE_gbm <- mean(abs(predict_gbm_backtransformed - y_var_test))

