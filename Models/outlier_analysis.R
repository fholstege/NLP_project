library(tidyverse)
library(smacof)
library(fastDummies)
library(ggrepel)
library(cluster)
library(factoextra)
library(arrow)

# load data
df <- read_csv('../data/clean/allInfo_min_CLS_GBM_prediction_all.csv')

# remove embeddings
df <- df %>% select(-starts_with('min_CLS'))

# merge with LDA topics
LDAdf <- read_parquet('../data/clean/all_journals_merged_topics_V1.gzip')

# keep only extracted topic proportions
LDAdf <- LDAdf %>% select(c(DOI, starts_with('Topic')))

# merge data frames base on DOI
df <- left_join(df, LDAdf, by = "DOI")

# make journal a factor and get as dummies
df$journal <- as.factor(df$journal)
df <- dummy_cols(df, select_columns = 'journal', remove_selected_columns = TRUE)

# replace NANs by 0
df[is.na(df)] <- 0

# compute errors
df$error <- df$citations - df$prediction_gbm_min_CLS

# only look at positive errors (citations > predicted citations)
df <- df %>% filter(error > 0)

# check distribution
ggplot(aes(x = error), data = df) + geom_density()

# get 95% quantile
q95 <- quantile(df$error, c(0.5, 0.8, 0.9, 0.95, 0.99))[4]

# determine outliers > 75% quantile + 1.5 IQR = 111 outliers
upper_bound <- quantile(df$error)[4] + 1.5*IQR(df$error)
upper_bound

# code outliers as 1
df$outlier_iqr <- ifelse(df$error > upper_bound, 1, 0) # 111 articles
df$outlier_q95 <- ifelse(df$error > q95, 1, 0) # 54 articles

# define dataset for table
regdf <- df %>% select(-c(DOI, prediction_gbm_min_CLS, dimensions_citations, error, refs_not_found, policy_mentions, outlier_q95)) %>% select(-starts_with('min_CLS'))

# summary stats table
library(RCT)
options(scipen=999)
table_df <- balance_table(regdf, "outlier_iqr")
names(table_df) <-  c('Variables', 'Non-Outlier', 'Outlier', 'PValue')
table_df

library(xtable)

xtable(table_df)


colnames(regdf)
test <- regdf[4:23]
rowSums(regdf[4:23])


D <- dist(regdf)
results <- mds(D)
plot(results)


sorted_df <- df %>% filter(outlier_iqr == 1) %>% arrange(desc(error)) %>% select(DOI, error, citations)
sorted_df <- df %>% filter(outlier_iqr == 1) %>% arrange(error) %>% select(DOI,  error, citations)



plotdf <- df %>% select(-c(DOI, prediction_gbm_min_CLS, dimensions_citations, refs_not_found, policy_mentions, outlier_q95)) %>% select(-starts_with('min_CLS'))
plotdf$outlier_group <- ifelse(plotdf$outlier_iqr == 1, "Outlier Article", "Non-Outlier Article")
ggplot(mapping = aes(x = citations, y = error, color = as.factor(outlier_group)), plotdf) + geom_point() + theme_bw() +
  xlab('Citations') + ylab('Citations - Predicted Citations (Error)')

