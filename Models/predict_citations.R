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
               pscl) 


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



df_word2vec_mean_representation = read.csv('../data/representations/word2vec_doc_representation.csv') %>% select(-X) %>% na.omit()
df_word2vec_max_representation = read.csv('../data/representations/word2vec_doc_representation.csv') %>% select(-X) %>% na.omit()
df_word2vec_min_representation = read.csv('../data/representations/word2vec_doc_representation.csv')%>% select(-X) %>% na.omit()


x_vars_mean <- model.matrix(num_ref~. , na.omit(df_word2vec_mean_representation))[,-1]
x_vars_max <-  model.matrix(num_ref~. , na.omit(df_word2vec_max_representation))[,-1]
x_vars_min <- model.matrix(num_ref~. , na.omit(df_word2vec_min_representation))[,-1]
y_var <- log(df_word2vec_mean_representation)

lambda_cv = 10^seq(2, -2, by = -0.5)
alpha_cv = seq(0,1,by=0.2)



# Splitting the data into test and train
set.seed(123)
train_indeces = which(df_word2vec_mean_representation$train_set == 1)

# mean representation
x_vars_mean_train = x_vars_mean[train_indeces,]
x_vars_mean_test = x_vars_mean[-train_indeces,]

# max representation
x_vars_max_train = x_vars_max[train_indeces,]
x_vars_max_test = x_vars_max[-train_indeces,]

# min representation
x_vars_min_train = x_vars_min[train_indeces,]
x_vars_min_test = x_vars_min[-train_indeces,]

# dependent variable
y_var_train = y_var[train_indeces]
y_var_test = y_var[-train_indeces]


# cv for mean representation
cv_mean_rep <- cv.glmnet(x_vars_mean_train, y_var_train,
                       alpha = alpha_cv, lambda = lambda_cv, 
                       nfolds = 5)

# cv for max representation
cv_mean_rep <- cv.glmnet(x_vars_max_train, y_var_train,
                         alpha = alpha_cv, lambda = lambda_cv, 
                         nfolds = 5)

cv_min_rep <- cv.glmnet(x_vars_min_train, y_var_train,
                         alpha = alpha_cv, lambda = lambda_cv, 
                         nfolds = 5)


opt_lambda_mean <- cv_output$lambda.min
opt_alpha_mean <- cv_output$alpha.min


# Rebuilding the model with best lamda value identified
lasso_model_mean_word2vec <- glmnet(x_vars_mean_train, y_var_train, alpha = opt_alpha_mean, lambda = opt_lambda_mean)
pred_elastic_mean <- predict(lasso_model_mean_word2vec, newx = x_vars_mean_test)
mean((pred - y_test)^2)

# add linear regression
