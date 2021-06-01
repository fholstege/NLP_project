##################################
# Author: Floris Holstege
# Goal: Predict the number of citations
#
# Parts: 
#   1) Load in packages, define help functions
#   2) Define dependent variable, log and winsorize it
#   3) Use topics from LDA
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
               glmnet) 


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
# 2) Define dependent variable, log and winsorize it
############################


# unlogged
ggplot(data = df_full,aes(x = `Cited by`) ) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations', y = 'Frequency') + 
  theme_bw()

# logged 
ggplot(data = df_full,aes(x = log(`Cited by`) )) + 
  geom_histogram(bins = 50, color="black", fill="blue") +
  labs(x = 'Number of Citations (Logarithm)', y = 'Frequency') + 
  theme_bw()

n_zero_citation = nrow(df_full[df_full$`Cited by`==0,])
n_zero_citation/ nrow(df_full)















# not add topic 3 - since all classified in one of these
topic_var = c('Topic 1', 'Topic 2', 'Topic 3')
dependent_var = c('num_ref')

df_model <- df[,c(dependent_var, topic_var)]
df_model_noNA <- na.omit(df_model)
colnames(df_model_noNA)

df_model_noNA

baseline_topic_model = lm(log(num_ref) ~ as.factor(Topic) , data = df_model_noNA)
summary(baseline_topic_model)




df_word2vec = read.csv('../data/representations/word2vec_doc_representation.csv')
df_word2vec <- df_word2vec %>% select(-X)
df_word2vec <- na.omit(df_word2vec)

x_vars <- model.matrix(num_ref~. , na.omit(df_word2vec))[,-1]
y_var <- log(df_word2vec$num_ref)
lambda_seq <- 10^seq(2, -2, by = -.1)


# Splitting the data into test and train
set.seed(123)
train = sample(1:nrow(x_vars), round(nrow(x_vars)*0.7,0))
x_test = (-train)
y_test = y_var[x_test]


sd(y_test)

cv_output <- cv.glmnet(x_vars[train,], y_var[train],
                       alpha = 1, lambda = lambda_seq, 
                       nfolds = 5)

opt_lambda <- cv_output$lambda.min
# Rebuilding the model with best lamda value identified
lasso_model_word2vec <- glmnet(x_vars[train,], y_var[train], alpha = 1, lambda = opt_lambda)
pred_lasso <- predict(lasso_model_word2vec, s = opt_lambda, newx = x_vars[x_test,])
mean((pred - y_test)^2)

baseline_word2vec_model = lm(y_var[train] ~ . , data = data.frame(x_vars[train,]))
pred_lm = predict(baseline_word2vec_model, newdata =  data.frame(x_vars[x_test,]))
mean((pred_lm - y_test)^2)
