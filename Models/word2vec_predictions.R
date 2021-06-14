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
               devtools,
               reshape2,
               stargazer,
               gridExtra,
               caTools) 


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
df_full = read_parquet('../data/clean/all_journals_merged_topics.gzip')
colnames(df_full)


############################
# 2) Define dependent variable
############################

# unlogged
citation_plot <- ggplot(data = df_full,aes(x = citations) ) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations', y = 'Frequency') + 
  theme_bw()+
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=20),
        axis.text.y = element_text(size=20),
        legend.text = element_text(size=14))

# logged (+1)
citation_plot_logged <- ggplot(data = df_full,aes(x = log(1 + citations) )) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations (Logarithm)', y = 'Frequency') + 
  theme_bw()+
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=20),
        axis.text.y = element_text(size=20),
        legend.text = element_text(size=14))
grid.arrange(citation_plot, citation_plot_logged, ncol=2)



# Altmetric 
altmetric_plot <- ggplot(data = df_full,aes(x = altmetric_score )) + 
  geom_histogram(bins = 20, color="black", fill="red") +
  labs(x = 'Altmetric Attention Score', y = 'Frequency') + 
  theme_bw()+ 
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=20),
        axis.text.y = element_text(size=20),
        legend.text = element_text(size=14))


# Altmetric  looged
altmetric_plot_logged <- ggplot(data = df_full,aes(x = log(1 + altmetric_score) )) + 
  geom_histogram(bins = 20, color="black", fill="red") +
  labs(x = 'Altmetric Attention Score (Logarithm) ', y = 'Frequency') + 
  theme_bw()+
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=20),
        axis.text.y = element_text(size=20),
        legend.text = element_text(size=14))

grid.arrange(altmetric_plot, altmetric_plot_logged, ncol=2)


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


## plot topics over time
topic_interpretations = c('Identity marketing','Luxury products', 'Customer Services', 'Advertising', 'Consumption', 'Digital Marketing', 'Store Promotion', 'Marketing Experiments', 'Business Marketing', 'Customer Relations')

# get which
list_name_topics <- lapply(df_full$Topic, function(x){
  
  name_topic = topic_interpretations[x]
  
  return(name_topic)
})
df_full$Topic_name <- unlist(list_name_topics)


df_topics <-  df_full %>%
  select('journal', 'year', 'DOI', 'Topic_name') %>%
  na.omit()


  


# topics over time
df_topics_time <- df_topics %>%
  group_by(year, Topic_name) %>%
  summarise(n_assigned =n()) %>%
  mutate(percent_assigned = n_assigned/sum(n_assigned))




# plot it 
ggplot(data = df_topics_time, aes(x = year, y = percent_assigned*100, fill = Topic_name))+
  geom_area()+
  theme_bw()+
  labs(x = 'Year', y = "% Of Papers Assigned To Topic", fill = 'Topic') + 
  scale_fill_brewer(palette="Set3")+
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=16),
        axis.text.y = element_text(size=16),
        legend.text = element_text(size=14))

# topics per journal
df_topics_journals <- df_topics %>%
  group_by(journal, Topic_name) %>%
  summarise(n_assigned =n()) %>%
  mutate(percent_assigned = n_assigned/sum(n_assigned)) %>%
  filter(journal != 'NULL')


df_topics_journals <- df_topics_journals %>%
  mutate(journal = paste0(journal))

# plot it 
ggplot(data = df_topics_journals, aes(y = percent_assigned*100, x=journal, fill = Topic_name))+
  geom_bar(position = 'stack', stat="identity") + 
  theme_bw()+
  labs(x = 'Topics', y = "% Of Papers Assigned To Topic", fill = 'Topic') +
  scale_fill_brewer(palette="Set3") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5)) + 
  theme(text = element_text(size=16), 
        axis.text.y = element_text(size=16),
        legend.text = element_text(size=14))




# removes a couple of empty rows
df_topics_model <- df_full %>%
  select('citations', 'altmetric_score','Topic_name', 'year', 'journal') 

df_citations_altmetric_topic <-  df_topics_model %>%
  group_by(Topic_name) %>%
  summarise(prob_cited = sum(ifelse(na.omit(citations) > 0, 1,0))/n(), 
            prob_altmetric = sum(ifelse(na.omit(altmetric_score) > 0, 1,0))/n()) %>%
  na.omit()

topic_most_likely_not = df_citations_altmetric_topic[which.min(df_citations_altmetric_topic$prob_cited),]$Topic_name[[1]]



# check for the entire dataset if the number of citations is zero
df_topics_model$citations_is_not_zero = ifelse(df_topics_model$citations > 0, 1, 0)


# create dataset where only citations are included that are not zero
df_model_non_zero_citations <- df_topics_model[df_topics_model$citations > 0,]




# run the binary, logit model for entire dataset
binary_model_topic = glm(citations_is_not_zero ~  relevel(as.factor(Topic_name), ref = topic_most_likely_not[[1]]) + year , data = na.omit(df_topics_model %>% select(-altmetric_score)), family = binomial(link = "logit"), control = list(maxit = 50))

# linear model for number of citations >0 
linear_model_topic = lm(log(1 + citations) ~ relevel(as.factor(Topic_name), ref = topic_most_likely_not[[1]]) + year   , data = na.omit(df_model_non_zero_citations%>% select(-altmetric_score)))

summary(binary_model_topic)
#coef_to_prob = function(coef){return (1/ (1 + exp(coef)))}
#coef_to_prob(binary_model_topic$coefficients)
summary(linear_model_topic)


## altmetric interest
df_topics_model$altmetric_is_zero = ifelse(df_topics_model$altmetric_score > 0, 1,0)
df_topics_non_zero_alt <- df_topics_model[df_topics_model$altmetric_score > 0,]
topic_most_likely_not = df_citations_altmetric_topic[which.min(df_citations_altmetric_topic$prob_altmetric),]$Topic_name[[1]]

df_citations_altmetric_topic
# run the binary, logit model for entire dataset
binary_model_topic_altmetric = glm(altmetric_is_zero ~ relevel(as.factor(Topic_name), ref = topic_most_likely_not[[1]]) + year , data = na.omit(df_topics_model %>% select(-citations)), family = binomial(link = "logit"), control = list(maxit = 50))

# linear model for number of altmetric >0 
linear_model_topic_altmetric = lm(log(1 + altmetric_score) ~relevel(as.factor(Topic_name), ref = topic_most_likely_not[[1]]) + year   , data = na.omit(df_topics_non_zero_alt%>% select(-citations)))

summary(binary_model_topic_altmetric)
summary(linear_model_topic_altmetric)

stargazer(binary_model_topic,binary_model_topic_altmetric,linear_model_topic,linear_model_topic_altmetric,digits=2)



############################
# 4) Word2vec
############################


# get dataframes 
df_word2vec_mean_representation = read_parquet('../data/representations/word2vec_doc_representation_mean.gzip') %>% na.omit()
df_word2vec_max_representation = read_parquet('../data/representations/word2vec_doc_representation_max.gzip')  %>% na.omit()
df_word2vec_min_representation = read_parquet('../data/representations/word2vec_doc_representation_min.gzip')%>% na.omit()
colnames(df_word2vec_mean_representation)

# get x and y values
x_vars_mean <- model.matrix(citations~. , na.omit(df_word2vec_mean_representation))[,-1]
x_vars_max <-  model.matrix(citations~. , na.omit(df_word2vec_max_representation))[,-1]
x_vars_min <- model.matrix(citations~. , na.omit(df_word2vec_min_representation))[,-1]
y_var <- df_word2vec_mean_representation$citations
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
K = 3

# function to speed up cv.glmnet in lapply with different alpha's 
cv_glmnet_wrapper <- function(alpha, x_vars, y_var, lambda_cv, K, ...){
  
  
  cv_result <- cv.glmnet(x_vars, 
            y_var,
            alpha = alpha, 
            lambda = lambda_cv, 
            nfolds = K,
            type.measure = 'mae',
            ...)
  
  return(cv_result)
}


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
cv_result_mean <- data.frame(list.rbind(lapply(cv_mean_rep, get_results_cv_wrapper)), alpha = alpha_cv)
colnames(cv_result_mean)[1:2] <- c('MAE', 'Lambda')
cv_result_mean
cv_result_max
which.min(cv_result_min$MAE)

# get cv results for max representation
cv_result_max <- data.frame(list.rbind(lapply(cv_max_rep, get_results_cv_wrapper)), alpha = alpha_cv)
colnames(cv_result_max)[1:2] <- c('MAE', 'Lambda')

# get cv results for min representation
cv_result_min <- data.frame(list.rbind(lapply(cv_min_rep, get_results_cv_wrapper)), alpha = alpha_cv)
colnames(cv_result_min)[1:2] <- c('MAE', 'Lambda')


# based on results, pick lambda and alpha
opt_alpha_mean = cv_result_mean[which.min(cv_result_mean$MAE),]$alpha
opt_lambda_mean = cv_result_mean[which.min(cv_result_mean$MAE),]$Lambda
opt_alpha_max = cv_result_max[which.min(cv_result_max$MAE),]$alpha
opt_lambda_max = cv_result_max[which.min(cv_result_max$MAE),]$Lambda
opt_alpha_min = cv_result_min[which.min(cv_result_min$MAE),]$alpha
opt_lambda_min =cv_result_min[which.min(cv_result_min$MAE),]$Lambda

# Rebuilding the model with best lamda value identified
elastic_model_word2vec_mean <- glmnet(as.matrix(x_vars_mean_train), y_var_train_logged, alpha = opt_alpha_mean, lambda = opt_lambda_mean)
elastic_model_word2vec_max <- glmnet(as.matrix(x_vars_max_train), y_var_train_logged, alpha = opt_alpha_max, lambda = opt_lambda_max)
elastic_model_word2vec_min <- glmnet(as.matrix(x_vars_min_train), y_var_train_logged, alpha = opt_alpha_min, lambda = opt_lambda_min)

# get predictions in log
pred_elastic_mean_log <- predict(elastic_model_word2vec_mean, newx = as.matrix(x_vars_mean_test))
pred_elastic_max_log <- predict(elastic_model_word2vec_max, newx = as.matrix(x_vars_max_test))
pred_elastic_min_log <- predict(elastic_model_word2vec_min, newx = as.matrix(x_vars_min_test))

# get the backtransformed predictions
pred_elastic_mean_backtransformed <- exp(pred_elastic_mean_log)-1
pred_elastic_max_backtransformed <- exp(pred_elastic_max_log)-1
pred_elastic_min_backtransformed <- exp(pred_elastic_min_log)-1

# use the elastic MAE
MAE_elastic_mean <- mean(abs(pred_elastic_mean_backtransformed - y_var_test))
MAE_elastic_max <- mean(abs(pred_elastic_max_backtransformed - y_var_test))
MAE_elastic_min <- mean(abs(pred_elastic_min_backtransformed - y_var_test))


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
  l_result_grid <- parallelsugar::mclapply(l_params ,gbm_inGridsearch,obj_formula = obj_formula, distribution = "gaussian", df_var = df_var, K = K, n_max_trees = n_max_trees,  mc.cores = detectCores()-2)
  
  # save results of the data.frame
  l_result_grid_clean <- lapply(l_result_grid, create_df_TreeResult)
  
  df_results <- do.call("rbind", l_result_grid_clean)
  
  return(df_results)
}

df_gridsearch_result_mean <- grid_search_gbm(y_var_train_logged ~ . , df_var =df_gbm_mean_rep_train, df_param , K, 10)

opt_index <- which.min(df_gridsearch_result_mean$cv_error)
opt_n_tree <- df_gridsearch_result_mean[opt_index, ]$n_tree
opt_min_obs <- df_gridsearch_result_mean[opt_index, ]$min_obs
opt_interaction_depth <- df_gridsearch_result_mean[opt_index, ]$interaction_depth


df_gridsearch_result_max <- grid_search_gbm(y_var_train_logged ~ . , df_var =df_gbm_max_rep_train, df_param , K, n_max_trees)
df_gridsearch_result_min <- grid_search_gbm(y_var_train_logged ~ . , df_var =df_gbm_min_rep_train, df_param , K, n_max_trees)


boosted_model_mean <- gbm(y_var_train_logged ~ . , data =df_gbm_mean_rep_train, distribution = 'gaussian', n.trees = 200, interaction.depth = 3, n.minobsinnode = 100)
boosted_model_max <- gbm(y_var_train_logged ~ . , data =df_gbm_max_rep_train, distribution = 'gaussian', n.trees = 230, interaction.depth = 3, n.minobsinnode = 100)
boosted_model_min <- gbm(y_var_train_logged ~ . , data =df_gbm_min_rep_train, distribution = 'gaussian', n.trees = 245, interaction.depth = 3, n.minobsinnode = 100)

df_gbm_mean_rep_test <-data.frame(y_var_test_logged, x_vars_mean_test)
df_gbm_max_rep_test <-data.frame(y_var_test_logged, x_vars_max_test)
df_gbm_min_rep_test <-data.frame(y_var_test_logged, x_vars_min_test)

pred_gbm_mean_logged <- predict(boosted_model_mean, df_gbm_mean_rep_test)
pred_gbm_max_logged <- predict(boosted_model_max, df_gbm_max_rep_test)
pred_gbm_min_logged <- predict(boosted_model_min, df_gbm_min_rep_test)


pred_gbm_mean_backtransformed <- exp(pred_gbm_mean_logged) + 1
pred_gbm_max_backtransformed <- exp(pred_gbm_max_logged) + 1
pred_gbm_min_backtransformed <- exp(pred_gbm_min_logged) + 1

MAE_gbm_mean <- mean(abs(pred_gbm_mean_backtransformed - y_var_test))
MAE_gbm_max <- mean(abs(pred_gbm_max_backtransformed - y_var_test))
MAE_gbm_min <- mean(abs(pred_gbm_min_backtransformed - y_var_test))

residual_MAE_gbm_max = abs(pred_gbm_max_backtransformed - y_var_test)
residual_MAE_elastic_net_

# plot residuals
df_residuals = data.frame(GBM = residual_MAE_gbm_max, 
                          elastic = abs(pred_elastic_min_backtransformed - y_var_test), 
                          linear =abs(pred_linear_model_max_backtransformed - y_var_test))

colnames(df_residuals) <- c('Gradient Boosting', 'Elastic Net', 'Linear Regression')
df_residuals_melted = melt(df_residuals)

ggplot(data = df_residuals_melted, aes(x = value, fill = variable)) + 
  geom_histogram( bins = 50) + 
  labs(x = 'Residuals', y = 'Frequency') + 
  theme_bw() +
  labs(fill = 'Type of model')+
  facet_grid(~variable)+
  theme(text = element_text(size=14))
