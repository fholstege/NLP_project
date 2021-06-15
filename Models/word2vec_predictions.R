##################################
# Author: Floris Holstege
# Goal: Predict the number of citations
#
# Parts: 
#   1) Load in packages, define help functions
#   2) Define dependent variable
#   3) check topics from LDA
#   4) Use word embeddings from Word2vec
#   5) combine information
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
               caTools,
               margins) 


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

# Citations
citation_plot <- ggplot(data = df_full,aes(x = citations) ) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations', y = 'Frequency') + 
  theme_bw()+
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=20),
        axis.text.y = element_text(size=20),
        legend.text = element_text(size=14))


# Altmetric 
altmetric_plot <- ggplot(data = df_full,aes(x = altmetric_score )) + 
  geom_histogram(bins = 20, color="black", fill="red") +
  labs(x = 'Altmetric Attention Score', y = 'Frequency') + 
  theme_bw()+ 
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=20),
        axis.text.y = element_text(size=20),
        legend.text = element_text(size=14))



grid.arrange(citation_plot,altmetric_plot, ncol=2)

# Use zero-inflated regression

############################
# 3) LDA topics
############################


## plot topics over time
topic_interpretations = c('Identity marketing','Luxury products', 'Customer Services', 'Advertising', 'Consumption', 'Digital Marketing', 'Store Promotion', 'Marketing Experiments', 'Business Marketing', 'Customer Relations')

# get which topic number is which topic name
list_name_topics <- lapply(df_full$Topic, function(x){
  
  name_topic = topic_interpretations[x]
  
  return(name_topic)
})

# change the variable
df_full$Topic_name <- unlist(list_name_topics)
colnames(df_full)[46:55] <- topic_interpretations

# create df for regression and analysis
df_topics <-  df_full %>%
  select('journal', 'year', 'DOI','Topic_name', topic_interpretations) %>%
  na.omit()
df_topics

df_topics_time <- df_topics %>%
  select(-journal, -DOI, -Topic_name)%>%
  melt(id.vars = 'year') %>%
  group_by(year, variable) %>%
  summarise(avg_prob = mean(value))
colnames(df_topics_time) <- c('year', 'Topic_name', 'avg_prob')


# plot it 
ggplot(data = df_topics_time, aes(x = year, y = avg_prob*100, fill = Topic_name))+
  geom_area()+
  theme_bw()+
  labs(x = 'Year', y = "Avg. Probability Of Being Assigned To Topic", fill = 'Topic') + 
  scale_fill_brewer(palette="Set3")+
  theme(text = element_text(size=16), 
        axis.text.x = element_text(size=16),
        axis.text.y = element_text(size=16),
        legend.text = element_text(size=14))

# topics per journal
df_topics_journals <- df_topics %>%
  select(-year, -DOI, -Topic_name)%>%
  melt(id.vars = 'journal') %>%
  group_by(journal, variable)%>%
  summarise(avg_prob =mean(value)) %>%
  filter(journal != 'NULL')
colnames(df_topics_journals) <- c('journal', 'Topic_name', 'avg_prob')

df_topics_journals <- df_topics_journals %>%
  mutate(journal = paste0(journal))
df_topics_journals

# plot it 
ggplot(data = df_topics_journals, aes(y = avg_prob*100, x=journal, fill = Topic_name))+
  geom_bar(position = 'stack', stat="identity") + 
  theme_bw()+
  labs(x = 'Topics', y = "Avg. Probability of being Assigned To Topic", fill = 'Topic') +
  scale_fill_brewer(palette="Set3") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5)) + 
  theme(text = element_text(size=16), 
        axis.text.y = element_text(size=16),
        legend.text = element_text(size=14))



df_full %>%
  select(DOI, citations, Topic_name) %>%
  melt(id.vars = c('DOI', 'citations')) %>%
  group_by(value) %>%
  summarise(avg_citations = mean(citations, na.rm = TRUE),
            n_zero_citations = sum(ifelse(citations == 0, 1,0), na.rm = TRUE),
            n_papers = n(),
            perc_zero_citations = n_zero_citations/n_papers)
  
  

# for hurdle with citations
df_topics_model_citations <- df_full %>%
  select('citations',topic_interpretations, 'year', 'journal') %>%
  na.omit() %>%
  select(-`Customer Relations`) # this is the baseline variable
df_topics_model_citations$year <- as.factor(df_topics_model_citations$year)
df_topics_model_citations$journal <- as.factor(paste0(df_topics_model_citations$journal))

# estimate both parts of the hurdle model for citations
binary_hurdle_citations = glm(ifelse(citations ==0, 1,0) ~ ., data = df_topics_model_citations, family = 'binomial')
poisson_hurdle_citations = glm(citations ~ ., data = hurdle_citations$model %>% filter(citations >0), family = 'poisson')

# get marginal effects for citations
margins_binary_hurdle_citations <- margins(binary_hurdle_citations)
margins_poisson_hurdle_citations <- margins(poisson_hurdle_citations)

df_AME_binary_citations = summary(margins_binary_hurdle_citations)%>% filter(factor %in% topic_interpretations)
df_AME_poisson_citations = summary(margins_poisson_hurdle_citations)%>% filter(factor %in% topic_interpretations)

# for hurdle with altmetric
df_topics_model_altmetric <- df_full %>%
  select('altmetric_score',topic_interpretations, 'year', 'journal') %>%
  na.omit() %>%
  select(-`Customer Relations`) # this is the baseline variable

df_topics_model_altmetric$year <- as.factor(df_topics_model_altmetric$year)
df_topics_model_altmetric$journal <- as.factor(paste0(df_topics_model_altmetric$journal))

# estimate both parts of the hurdle model for citations
binary_hurdle_altmetric = glm(ifelse(altmetric_score ==0, 1,0) ~ ., data = df_topics_model_altmetric, family = 'binomial')
poisson_hurdle_altmetric = glm(altmetric_score ~ ., data = df_topics_model_altmetric %>% filter(altmetric_score >0), family = 'poisson')

# get marginal effects for citations
margins_binary_hurdle_altmetric <- margins(binary_hurdle_altmetric)
margins_poisson_hurdle_altmetric <- margins(poisson_hurdle_altmetric)

df_AME_binary_altmetric = summary(margins_binary_hurdle_altmetric) %>% filter(factor %in% topic_interpretations)
df_AME_poisson_altmetric = summary(margins_poisson_hurdle_altmetric) %>% filter(factor %in% topic_interpretations)


############################
# 4) Word2vec
############################
getwd()

# get dataframes 
df_word2vec_mean_300_representation = read_parquet('../data/representations/word2vec_doc_representation_300_mean.gzip') %>% na.omit()
df_word2vec_max_300_representation = read_parquet('../data/representations/word2vec_doc_representation_300_max.gzip')  %>% na.omit()
df_word2vec_min_300_representation = read_parquet('../data/representations/word2vec_doc_representation_300_min.gzip')%>% na.omit()
df_word2vec_mean_500_representation = read_parquet('../data/representations/word2vec_doc_representation_500_mean.gzip') %>% na.omit()
df_word2vec_max_500_representation = read_parquet('../data/representations/word2vec_doc_representation_500_max.gzip')  %>% na.omit()
df_word2vec_min_500_representation = read_parquet('../data/representations/word2vec_doc_representation_500_min.gzip')%>% na.omit()

# get x and y values
x_vars_mean_300 <- model.matrix(citations~. , na.omit(df_word2vec_mean_300_representation) %>% select( -DOI))[,-1]
x_vars_max_300 <-  model.matrix(citations~. , na.omit(df_word2vec_max_300_representation) %>% select( -DOI))[,-1]
x_vars_min_300 <- model.matrix(citations~. , na.omit(df_word2vec_min_300_representation) %>% select( -DOI))[,-1]
x_vars_mean_500 <- model.matrix(citations~. , na.omit(df_word2vec_mean_500_representation) %>% select( -DOI))[,-1]
x_vars_max_500 <-  model.matrix(citations~. , na.omit(df_word2vec_max_500_representation) %>% select( -DOI))[,-1]
x_vars_min_500 <- model.matrix(citations~. , na.omit(df_word2vec_min_500_representation)%>% select(-DOI))[,-1]
y_var <- df_word2vec_mean_300_representation$citations %>% na.omit()

# try out these values for alpha and lambda
lambda_cv =  c( 0.1, 1, 10,100)
alpha_cv = seq(0,1,by=0.5)

# Splitting the data into test and train
set.seed(123)
train_indeces = which(df_word2vec_mean_300_representation$train_set == 1)


# mean representation
x_vars_mean_300_train = data.frame(x_vars_mean_300[train_indeces,]) %>% select(-train_set)
x_vars_mean_300_test = data.frame(x_vars_mean_300[-train_indeces,])  %>% select(-train_set)
x_vars_mean_500_train = data.frame(x_vars_mean_500[train_indeces,]) %>% select(-train_set)
x_vars_mean_500_test = data.frame(x_vars_mean_500[-train_indeces,])  %>% select(-train_set)

# max representation
x_vars_max_300_train = data.frame(x_vars_max_300[train_indeces,]) %>% select(-train_set)
x_vars_max_300_test = data.frame(x_vars_max_300[-train_indeces,])  %>% select(-train_set)
x_vars_max_500_train = data.frame(x_vars_max_500[train_indeces,]) %>% select(-train_set)
x_vars_max_500_test = data.frame(x_vars_max_500[-train_indeces,])  %>% select(-train_set)

# min representation
x_vars_min_300_train = data.frame(x_vars_min_300[train_indeces,]) %>% select(-train_set)
x_vars_min_300_test = data.frame(x_vars_min_300[-train_indeces,])  %>% select(-train_set)
x_vars_min_500_train = data.frame(x_vars_min_500[train_indeces,]) %>% select(-train_set)
x_vars_min_500_test = data.frame(x_vars_min_500[-train_indeces,])  %>% select(-train_set)

# dependent variable
y_var_train = y_var[train_indeces]
y_var_test = y_var[-train_indeces]

####
# First; baseline model with linear regression
####

# get the models
linear_model_mean_rep_300 <- lm(y_var_train ~ . , data = data.frame(x_vars_mean_300_train))
linear_model_max_rep_300 <- lm(y_var_train ~ ., data = data.frame(x_vars_max_300_train))
linear_model_min_rep_300 <- lm(y_var_train ~ ., data = data.frame(x_vars_min_300_train))
linear_model_mean_rep_500 <- lm(y_var_train ~ . , data = data.frame(x_vars_mean_500_train))
linear_model_max_rep_500 <- lm(y_var_train ~ ., data = data.frame(x_vars_max_500_train))
linear_model_min_rep_500 <- lm(y_var_train ~ ., data = data.frame(x_vars_min_500_train))

# get the predictions
pred_linear_model_mean_300 <- predict(linear_model_mean_rep_300, data.frame(x_vars_mean_300_test))
pred_linear_model_max_300 <- predict(linear_model_max_rep_300, data.frame(x_vars_max_300_test))
pred_linear_model_min_300 <- predict(linear_model_min_rep_300, data.frame(x_vars_min_300_test))
pred_linear_model_mean_500 <- predict(linear_model_mean_rep_500, data.frame(x_vars_mean_500_test))
pred_linear_model_max_500 <- predict(linear_model_max_rep_500, data.frame(x_vars_max_500_test))
pred_linear_model_min_500 <- predict(linear_model_min_rep_500, data.frame(x_vars_min_500_test))


# get the MSE
MSE_linear_model_mean_300 <- mean((pred_linear_model_mean_300 - y_var_test)^2)
MSE_linear_model_max_300 <- mean((pred_linear_model_max_300 - y_var_test)^2)
MSE_linear_model_min_300 <- mean((pred_linear_model_min_300 - y_var_test)^2)
MSE_linear_model_mean_500 <- mean((pred_linear_model_mean_500 - y_var_test)^2)
MSE_linear_model_max_500 <- mean((pred_linear_model_max_500 - y_var_test)^2)
MSE_linear_model_min_500 <- mean((pred_linear_model_min_500 - y_var_test)^2)


# get the MAE
MAE_linear_model_mean_300 <- mean(abs(pred_linear_model_mean_300 - y_var_test))
MAE_linear_model_max_300 <- mean(abs(pred_linear_model_max_300 - y_var_test))
MAE_linear_model_min_300 <- mean(abs(pred_linear_model_min_300 - y_var_test))
MAE_linear_model_mean_500 <- mean(abs(pred_linear_model_mean_500 - y_var_test))
MAE_linear_model_max_500 <- mean(abs(pred_linear_model_max_500 - y_var_test))
MAE_linear_model_min_500 <- mean(abs(pred_linear_model_min_500 - y_var_test))




####
# Second; elastic net
####


# number of folds
K = 4

# function to speed up cv.glmnet in lapply with different alpha's 
cv_glmnet_wrapper <- function(alpha, x_vars, y_var, lambda_cv, K, ...){
  
  
  cv_result <- cv.glmnet(x_vars, 
            y_var,
            alpha = alpha, 
            lambda = lambda_cv, 
            nfolds = K,
            type.measure = 'mse',
            ...)
  
  return(cv_result)
}


# cv for mean representation
cv_mean_rep_300 <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_mean_300_train), y = y_var_train, lambda_cv = lambda_cv, K=K)
cv_mean_rep_500 <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_mean_500_train), y = y_var_train, lambda_cv = lambda_cv, K=K)

# cv for max representation
cv_max_rep_300 <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_max_300_train), y = y_var_train, lambda_cv = lambda_cv, K=K)
cv_max_rep_500 <- lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_max_500_train), y = y_var_train, lambda_cv = lambda_cv, K=K)


# cv for min representation
cv_min_rep_300 <-lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_min_300_train), y = y_var_train, lambda_cv = lambda_cv, K=K)
cv_min_rep_500 <-lapply(alpha_cv, cv_glmnet_wrapper, x_vars= as.matrix(x_vars_min_500_train), y = y_var_train, lambda_cv = lambda_cv, K=K)


# function to get results from the wrapper function
get_results_cv_wrapper <- function(x){ 
  score = min(x$cvm)
  lambda = x$lambda.min
  
  return(c(score, lambda))
  }

# get cv results for mean representation
cv_result_mean_300 <- data.frame(list.rbind(lapply(cv_mean_rep_300, get_results_cv_wrapper)), alpha = alpha_cv)
cv_result_mean_500 <- data.frame(list.rbind(lapply(cv_mean_rep_500, get_results_cv_wrapper)), alpha = alpha_cv)
colnames(cv_result_mean_300)[1:2] <- c('MSE', 'Lambda')
colnames(cv_result_mean_500)[1:2] <- c('MSE', 'Lambda')

# get cv results for max representation
cv_result_max_300 <- data.frame(list.rbind(lapply(cv_max_rep_300, get_results_cv_wrapper)), alpha = alpha_cv)
cv_result_max_500 <- data.frame(list.rbind(lapply(cv_max_rep_500, get_results_cv_wrapper)), alpha = alpha_cv)
colnames(cv_result_max_300)[1:2] <- c('MSE', 'Lambda')
colnames(cv_result_max_500)[1:2] <- c('MSE', 'Lambda')

# get cv results for min representation
cv_result_min_300 <- data.frame(list.rbind(lapply(cv_min_rep_300, get_results_cv_wrapper)), alpha = alpha_cv)
cv_result_min_500 <- data.frame(list.rbind(lapply(cv_min_rep_500, get_results_cv_wrapper)), alpha = alpha_cv)
colnames(cv_result_min_300)[1:2] <- c('MSE', 'Lambda')
colnames(cv_result_min_500)[1:2] <- c('MSE', 'Lambda')


## based on results, pick lambda and alpha
# mean
opt_alpha_mean_300 = cv_result_mean_300[which.min(cv_result_mean_300$MSE),]$alpha
opt_lambda_mean_300 = cv_result_mean_300[which.min(cv_result_mean_300$MSE),]$Lambda
opt_alpha_mean_500 = cv_result_mean_500[which.min(cv_result_mean_500$MSE),]$alpha
opt_lambda_mean_500 = cv_result_mean_500[which.min(cv_result_mean_500$MSE),]$Lambda

# max
opt_alpha_max_300 = cv_result_max_300[which.min(cv_result_max_300$MSE),]$alpha
opt_lambda_max_300 = cv_result_max_300[which.min(cv_result_max_300$MSE),]$Lambda
opt_alpha_max_500 = cv_result_max_500[which.min(cv_result_max_500$MSE),]$alpha
opt_lambda_max_500 = cv_result_max_500[which.min(cv_result_max_500$MSE),]$Lambda

# min
opt_alpha_min_300 = cv_result_min_300[which.min(cv_result_min_300$MSE),]$alpha
opt_lambda_min_300 = cv_result_min_300[which.min(cv_result_min_300$MSE),]$Lambda
opt_alpha_min_500 = cv_result_min_500[which.min(cv_result_min_500$MSE),]$alpha
opt_lambda_min_500 = cv_result_min_500[which.min(cv_result_min_500$MSE),]$Lambda




# Rebuilding the model with best lamda value identified
elastic_model_word2vec_mean_300 <- glmnet(as.matrix(x_vars_mean_300_train), y_var_train, alpha = opt_alpha_mean_300, lambda = opt_lambda_mean_300)
elastic_model_word2vec_max_300 <- glmnet(as.matrix(x_vars_max_300_train), y_var_train, alpha = opt_alpha_max_300, lambda = opt_lambda_max_300)
elastic_model_word2vec_min_300 <- glmnet(as.matrix(x_vars_min_300_train), y_var_train, alpha = opt_alpha_min_300, lambda = opt_lambda_min_300)
elastic_model_word2vec_mean_500 <- glmnet(as.matrix(x_vars_mean_500_train), y_var_train, alpha = opt_alpha_mean_500, lambda = opt_lambda_mean_500)
elastic_model_word2vec_max_500 <- glmnet(as.matrix(x_vars_max_500_train), y_var_train, alpha = opt_alpha_max_500, lambda = opt_lambda_max_500)
elastic_model_word2vec_min_500 <- glmnet(as.matrix(x_vars_min_500_train), y_var_train, alpha = opt_alpha_min_500, lambda = opt_lambda_min_500)


# get predictions i
pred_elastic_mean_300 <- predict(elastic_model_word2vec_mean_300, newx = as.matrix(x_vars_mean_300_test))
pred_elastic_max_300 <- predict(elastic_model_word2vec_max_300, newx = as.matrix(x_vars_max_300_test))
pred_elastic_min_300 <- predict(elastic_model_word2vec_min_300, newx = as.matrix(x_vars_min_300_test))
pred_elastic_mean_500 <- predict(elastic_model_word2vec_mean_500, newx = as.matrix(x_vars_mean_500_test))
pred_elastic_max_500 <- predict(elastic_model_word2vec_max_500, newx = as.matrix(x_vars_max_500_test))
pred_elastic_min_500 <- predict(elastic_model_word2vec_min_500, newx = as.matrix(x_vars_min_500_test))

# use the elastic MSE
MSE_elastic_mean_300 <- mean((pred_elastic_mean_300 - y_var_test)^2)
MSE_elastic_max_300 <- mean((pred_elastic_max_300 - y_var_test)^2)
MSE_elastic_min_300 <- mean((pred_elastic_min_300 - y_var_test)^2)
MSE_elastic_mean_500 <- mean((pred_elastic_mean_500 - y_var_test)^2)
MSE_elastic_max_500 <- mean((pred_elastic_max_500 - y_var_test)^2)
MSE_elastic_min_500 <- mean((pred_elastic_min_500 - y_var_test)^2)


# use the elastic MAE
MAE_elastic_mean_300 <- mean(abs(pred_elastic_mean_300 - y_var_test))
MAE_elastic_max_300 <- mean(abs(pred_elastic_max_300 - y_var_test))
MAE_elastic_min_300 <- mean(abs(pred_elastic_min_300 - y_var_test))
MAE_elastic_mean_500 <- mean(abs(pred_elastic_mean_500 - y_var_test))
MAE_elastic_max_500 <- mean(abs(pred_elastic_max_500 - y_var_test))
MAE_elastic_min_500 <- mean(abs(pred_elastic_min_500 - y_var_test))


####
# Third; random forest 
####

df_gbm_mean_rep_train_300 = data.frame(y_var_train, x_vars_mean_300_train)
df_gbm_max_rep_train_300 = data.frame(y_var_train, x_vars_max_300_train)
df_gbm_min_rep_train_300 = data.frame(y_var_train, x_vars_min_300_train)
df_gbm_mean_rep_train_500 = data.frame(y_var_train, x_vars_mean_500_train)
df_gbm_max_rep_train_500 = data.frame(y_var_train, x_vars_max_500_train)
df_gbm_min_rep_train_500 = data.frame(y_var_train, x_vars_min_500_train)

n_max_trees = 200
interaction_depth_seq = c(1,3)
n_min_obs_node_seq = c(100,200)

df_param <- expand.grid(interaction_depth_seq, n_min_obs_node_seq)
colnames(df_param) <- c('interaction_depth', 'min_obs')


# ensures we can apply gbm in lapply
gbm_inGridsearch <- function(l_param, obj_formula, distribution, df_var, K, n_max_trees){

    result <- gbm(formula = obj_formula, 
                distribution = distribution, 
                data = df_var, 
                cv.folds = K, 
                n.trees = n_max_trees, 
                interaction.depth = l_param$interaction_depth,
                n.minobsinnode = l_param$min_obs, 
                n.cores = 4)
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
  l_result_grid <- lapply(l_params ,gbm_inGridsearch,obj_formula = obj_formula, distribution = "gaussian", df_var = df_var, K = K, n_max_trees = n_max_trees)
  
  # save results of the data.frame
  l_result_grid_clean <- lapply(l_result_grid, create_df_TreeResult)
  
  df_results <- do.call("rbind", l_result_grid_clean)
  
  return(df_results)
}

# gridsearch
df_gridsearch_result_mean_300 <- grid_search_gbm(y_var_train ~ . , df_var =df_gbm_mean_rep_train_300, df_param = df_param , K=K, n_max_trees=n_max_trees)
df_gridsearch_result_max_300 <- grid_search_gbm(y_var_train ~ . , df_var =df_gbm_max_rep_train_300, df_param = df_param, K = K, n_max_trees = n_max_trees)
df_gridsearch_result_min_300 <- grid_search_gbm(y_var_train ~ . , df_var =df_gbm_min_rep_train_300, df_param , K, n_max_trees)
df_gridsearch_result_mean_500 <- grid_search_gbm(y_var_train ~ . , df_var =df_gbm_mean_rep_train_500, df_param = df_param , K=K, n_max_trees=n_max_trees)
df_gridsearch_result_max_500 <- grid_search_gbm(y_var_train ~ . , df_var =df_gbm_max_rep_train_500, df_param = df_param, K = K, n_max_trees = n_max_trees)
df_gridsearch_result_min_500 <- grid_search_gbm(y_var_train ~ . , df_var =df_gbm_min_rep_train_500, df_param , K, n_max_trees)



# boosted models - FILL IN PARAMETERS
boosted_model_mean_300 <- gbm(y_var_train ~ . , data =df_gbm_mean_rep_train_300, distribution = 'gaussian', n.trees = 195, interaction.depth = 3, n.minobsinnode = 100)
boosted_model_max_300 <- gbm(y_var_train ~ . , data =df_gbm_max_rep_train_300, distribution = 'gaussian', n.trees = 195, interaction.depth = 3, n.minobsinnode = 100)
boosted_model_min_300 <- gbm(y_var_train ~ . , data =df_gbm_min_rep_train_300, distribution = 'gaussian', n.trees = 120, interaction.depth = 3, n.minobsinnode = 100)
boosted_model_mean_500 <- gbm(y_var_train ~ . , data =df_gbm_mean_rep_train_500, distribution = 'gaussian', n.trees = 200, interaction.depth = 3, n.minobsinnode = 200)
boosted_model_max_500 <- gbm(y_var_train ~ . , data =df_gbm_max_rep_train_500, distribution = 'gaussian', n.trees = 100, interaction.depth = 3, n.minobsinnode = 100)
boosted_model_min_500 <- gbm(y_var_train ~ . , data =df_gbm_min_rep_train_500, distribution = 'gaussian', n.trees = 60, interaction.depth = 3, n.minobsinnode = 100)

# get the test dataframe
df_gbm_mean_rep_test_300 <-data.frame(y_var_test, x_vars_mean_300_test)
df_gbm_max_rep_test_300 <-data.frame(y_var_test, x_vars_max_300_test)
df_gbm_min_rep_test_300 <-data.frame(y_var_test, x_vars_min_300_test)
df_gbm_mean_rep_test_500 <-data.frame(y_var_test, x_vars_mean_500_test)
df_gbm_max_rep_test_500 <-data.frame(y_var_test, x_vars_max_500_test)
df_gbm_min_rep_test_500 <-data.frame(y_var_test, x_vars_min_500_test)

# get the predictions
pred_gbm_mean_300 <- predict(boosted_model_mean_300, df_gbm_mean_rep_test_300)
pred_gbm_max_300 <- predict(boosted_model_max_300, df_gbm_max_rep_test_300)
pred_gbm_min_300 <- predict(boosted_model_min_300, df_gbm_min_rep_test_300)
pred_gbm_mean_500 <- predict(boosted_model_mean_500, df_gbm_mean_rep_test_500)
pred_gbm_max_500 <- predict(boosted_model_max_500, df_gbm_max_rep_test_500)
pred_gbm_min_500 <- predict(boosted_model_min_500, df_gbm_min_rep_test_500)

# MSE and MAE
MSE_gbm_mean_300 <- mean((pred_gbm_mean_300 - y_var_test)^2)
MSE_gbm_max_300 <- mean((pred_gbm_max_300 - y_var_test)^2)
MSE_gbm_min_300 <- mean((pred_gbm_min_300 - y_var_test)^2)
MSE_gbm_mean_500 <- mean((pred_gbm_mean_500 - y_var_test)^2)
MSE_gbm_max_500 <- mean((pred_gbm_max_500 - y_var_test)^2)
MSE_gbm_min_500 <- mean((pred_gbm_min_500 - y_var_test)^2)

MAE_gbm_mean_300 <- mean(abs(pred_gbm_mean_300 - y_var_test))
MAE_gbm_max_300 <- mean(abs(pred_gbm_max_300 - y_var_test))
MAE_gbm_min_300 <- mean(abs(pred_gbm_min_300 - y_var_test))
MAE_gbm_mean_500 <- mean(abs(pred_gbm_mean_500 - y_var_test))
MAE_gbm_max_500 <- mean(abs(pred_gbm_max_500 - y_var_test))
MAE_gbm_min_500 <- mean(abs(pred_gbm_min_500 - y_var_test))



df_result = data.frame(MSE = c(MSE_gbm_mean_300, MSE_gbm_max_300, MSE_gbm_min_300, MSE_gbm_mean_500, MSE_gbm_max_500, MSE_gbm_min_500),
                       MAE = c(MAE_gbm_mean_300, MAE_gbm_max_300, MAE_gbm_min_300, MAE_gbm_mean_500, MAE_gbm_max_500, MAE_gbm_min_500))

round(df_result,1)

# plot residuals
df_residuals = data.frame(GBM = abs(pred_gbm_mean - y_var_test), 
                          elastic = abs(pred_elastic_mean - y_var_test), 
                          linear =abs(pred_linear_model_mean - y_var_test))

df_full_residuals = cbind(df_full[-train_indeces,] %>% filter(!is.na(citations) & !is.na(year)),df_residuals)

colnames(df_residuals) <- c('Gradient Boosting', 'Elastic Net', 'Linear Regression')
df_residuals_melted = melt(df_residuals)

ggplot(data = df_residuals_melted, aes(x = value, fill = variable)) + 
  geom_histogram( bins = 50) + 
  labs(x = 'Residuals', y = 'Frequency') + 
  theme_bw() +
  labs(fill = 'Type of model')+
  facet_grid(~variable)+
  theme(text = element_text(size=14))



