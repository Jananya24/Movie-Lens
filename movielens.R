#Installing the required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

#Attaching the required packages
library(tidyverse)
library(caret)
library(data.table)
library(reshape2)
library(lubridate)
library(dplyr) 
library(Rcpp)
library(ggplot2)
library(ggthemes)

#Loading the dataset
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId), title = as.character(title), genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

#DATA PREPARATION

#Validation set: 10% of the data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

#Train set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in train set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(test_index, temp, removed)


#DATA EXPLORATION

str(edx)
dim(edx)
names(edx)

head(edx)
summary(edx)
sd(edx$rating)
summary(edx$rating)

tapply(edx$rating, edx$genres, sum)
tapply(edx$rating, edx$genres, mean)

tot_observation <- length(edx$rating) + length(validation$rating) 
tot_observation
edx %>% group_by(genres) %>% summarise(n=n()) %>% head()
tibble(count = str_count(edx$genres, fixed("|")), genres = edx$genres) %>% group_by(count, genres) %>% summarise(n = n()) %>% arrange(-count) %>% head()

#DATA VISUALISATION

# Ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

# Number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

# Ratings Users - Mean
edx %>%
  group_by(userId) %>%
  summarise(mu_user = mean(rating)) %>%
  ggplot(aes(mu_user)) +
  geom_histogram(color = "black") +
  ggtitle("Average Ratings per User Histogram") +
  xlab("Average Rating") +
  ylab("# User") +
  theme(plot.title = element_text(hjust = 0.5))

#number of rating for each movie genres
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>% ggplot(aes(genres,count)) + 
  geom_bar(aes(fill =genres),stat = "identity")+ 
  labs(title = " Number of Rating for Each Genre")+
  theme(axis.text.x  = element_text(angle= 90, vjust = 50 ))+
  theme_light()

#Modelling Approaches

#RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Average Ratings Model
mu <- mean(edx$rating)
mu
model_1 <- RMSE(validation$rating, mu)
naive_rmse
rmse_results <- data_frame(Model = "Average", RMSE = model_1)
rmse_results


#MOVIE AND USERS MODEL

#Cross validation to determine lambda

lambdas <- seq(0, 8, 0.1)
rmses_2 <- matrix(nrow=10,ncol=length(lambdas))
# perform 10-fold cross validation to determine the optimal lambda
for(k in 1:10) {
  train_set <- edx[cv_splits[[k]],]
  test_set <- edx[-cv_splits[[k]],]
  
  # Make sure userId and movieId in test set are also in the train set
  test_final <- test_set %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(test_set, test_final)
  train_final <- rbind(train_set, removed)
  
  mu <- mean(train_final$rating)
  
  rmses_2[k,] <- sapply(lambdas, function(l){
    b_i <- train_final %>% 
      group_by(movieId) %>%
      summarize(b_i = sum(rating - mu)/(n()+l))
    b_u <- train_final %>% 
      left_join(b_i, by="movieId") %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - b_i - mu)/(n()+l))
    predicted_ratings <- 
      test_final %>% 
      left_join(b_i, by = "movieId") %>%
      left_join(b_u, by = "userId") %>%
      mutate(pred = mu + b_i + b_u) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_final$rating))
  })
}

rmses_2
rmses_2_cv <- colMeans(rmses_2)
rmses_2_cv
qplot(lambdas,rmses_2_cv)
lambda <- lambdas[which.min(rmses_2_cv)]    

mu <- mean(edx$rating)
b_i_reg <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u_reg <- edx %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted_ratings_6 <- 
  validation %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_6_rmse <- RMSE(predicted_ratings_6, validation$rating)   
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Regularized Movie + User Effect Model",  
                                     RMSE = model_6_rmse))
rmse_results 

#MOVIE + USER + WEEK
lambda = 0.75
week_avg_reg <- test_set %>% 
  group_by(week) %>% 
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  summarise(n=n(), w_ui_reg = sum(rating - mu - b_i - b_u) / (lambda + n) )

w_ui_reg <- test_set %>% 
  mutate(date = as_datetime(timestamp), 
         week = round_date(date, unit="week")) %>%
  left_join(movie_avg, by="movieId") %>%
  left_join(user_avg, by="userId") %>%
  left_join(week_avg_reg, by="week") %>%
  .$w_ui_reg
# Now there's only 1 NA when using week effect. Replace NA with mu
w_ui_reg <- replace_na(w_ui_reg, mu)
y_hat <- mu + b_i + b_u + w_ui_reg
model_6_rmse <- rmse(test_set$rating, y_hat)


lambdas <- seq(0, 5, 0.25)
rmses <- sapply(lambdas, function(lambda){
  movie_avg_reg <- train_set %>%
    group_by(movieId) %>% 
    summarise(n=n(), b_i_reg = sum(rating - mu) / (lambda + n) )
  
  user_avg_reg <- train_set %>%
    group_by(userId) %>% 
    left_join(movie_avg, by="movieId") %>%
    summarise(n=n(), b_u_reg = sum(rating - mu - b_i) / (lambda + n) )
  
  b_i_reg <- test_set %>% 
    left_join(movie_avg_reg, by="movieId") %>%
    .$b_i_reg
  
  b_u_reg <- test_set %>% 
    left_join(user_avg_reg, by="userId") %>%
    .$b_u_reg
  
  y_hat <- mu + b_i_reg + b_u_reg
  
  return(rmse(y_hat, test_set$rating))
})


#Results

rmse_results <- data.frame(Method = c("Simple Model", 
                                      "Movie + User Effect", 
                                      "Regularized Movie + User + Week Effect"), 
                           RMSE = c(model_1, model_2, model_3))
rmse_results %>% knitr::kable()