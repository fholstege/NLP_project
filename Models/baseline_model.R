getwd()


library(reticulate)
library(tidyverse)
library(glmnet)


#py_install('pyarrow')
pandas <- import("pandas")
read_parquet <- function(path, columns = NULL) {
  
  path <- path.expand(path)
  path <- normalizePath(path)
  
  if (!is.null(columns)) columns = as.list(columns)
  
  xdf <- pandas$read_parquet(path, columns = columns)
  
  xdf <- as.data.frame(xdf, stringsAsFactors = FALSE)
  
  dplyr::tbl_df(xdf)
  
}

df = read_parquet('../data/clean/data_journals_merged_topics.gzip')

colnames(df)

# not add topic 3 - since all classified in one of these
independent_var = c('Topic')
dependent_var = c('num_ref')

df_model <- df[,c(dependent_var, independent_var)]
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
