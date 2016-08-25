
library(glmnet)
library(xgboost)
library(ROCR)


## Generate simulated data 
## modified Hastie etal 10.2 with x_i>0 (1st quadrant in 2D)

genr_data <- function(n,p) { 
  X <- matrix(abs(rnorm(n*p)),n,p)
  y <- apply(X,1, function(x) ifelse(sum(x^2) > qchisq(0.5,p),1,0))  
  list(X = X,y = y)
}


## Compute AUC on test data

get_auc <- function(md, d_test, ...) {
  phat <- predict(md, d_test$X, ...)
  rocr_pred <- prediction(phat, d_test$y)
  performance(rocr_pred, "auc")@y.values[[1]]
}



## Generate data

set.seed(123)

n <- 2000
p <- 10

d_train <- genr_data(n,p)
d_valid <- genr_data(10000,p)
d_test  <- genr_data(10000,p)



## Logistic regression

system.time({
  md <- glmnet(d_train$X, d_train$y, family = "binomial", lambda = 0)
})

get_auc(md, d_test, type = "response")



## GBM

dxgb_train <- xgb.DMatrix(data = d_train$X, label = d_train$y)
dxgb_valid <- xgb.DMatrix(data = d_valid$X, label = d_valid$y)
dxgb_test  <- xgb.DMatrix(data = d_test$X,  label = d_test$y)

system.time({
md <- xgb.train(data = dxgb_train, nthread = parallel::detectCores(), 
            objective = "binary:logistic", nrounds = 1000, 
            max_depth = 10, eta = 0.1,
            watchlist = list(valid = dxgb_valid, train = dxgb_train), eval_metric = "auc",
            early.stop.round = 50, print.every.n = 50)
})

get_auc(md, d_test)






