install.packages("Metrics")
install.packages("ggplot2")
library("ggplot2")
library("Metrics")


df_pred<-data.frame()

#Data
df<-read.csv("C:\\FHWn\\Master-AGES\\8_Data\\complete_data.csv")
head(df)

#take away the python rownumber
df<-subset(df, select=-c(X))

#Scale DF
df<-scale(df, TRUE, TRUE)
summary(df)



#make train and test df

smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

x_train_df <- data.frame(df[train_ind, ])
y_train<-x_train_df$RefSt
#x_train_df = subset(x_train_df, select = - RefSt)

x_test_df <- data.frame(df[-train_ind, ])
y_test<-x_test_df$RefSt
#x_test_df = subset(x_test_df, select = - RefSt )


#MLR

model_MLR <- lm(RefSt ~., data=x_train_df)
summary(model_MLR)

    #Metrics
rmse(y_train, fitted(model_MLR)) #0.5
Metrics::mse(y_train, fitted(model_MLR)) #0.255
Metrics::mae(y_train, fitted(model_MLR)) #0.350

  #Test Data
y_hat<-predict(model_MLR,newdata = x_test_df)

rmse(y_test, y_hat) #0.53
Metrics::mse(y_test, y_hat) #0.287
Metrics::mae(y_test, y_hat) #0.36
Metrics::rse(y_test, y_hat) #0.25


a <- ggplot(x_test_df, aes(x = RefSt, y = Sensor))
a <- a + geom_point()
a <- a + geom_smooth(method = "loess", se = FALSE)
a <- a + geom_smooth(method = "lm", se = FALSE, color = "red")
a


x = 1:length(y_test)

plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, y_hat, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()


#Ridge Regression

smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

x_train_df <- data.frame(df[train_ind, ])
y_train<-x_train_df$RefSt
x_train_df = subset(x_train_df, select = - RefSt)

x_test_df <- data.frame(df[-train_ind, ])
y_test<-x_test_df$RefSt
x_test_df = subset(x_test_df, select = - RefSt )


library(glmnet)
install.packages("shape")
library(shape)

lambda <- 10^seq(2, -3, by = -.1)
ridge_reg = glmnet(x_train_df, y_train, nlambda = 25, alpha = 0, family = 'gaussian', lambda = 0.001)
summary(ridge_reg)


  # Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))


  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )

}

predictions_train <- predict(ridge_reg, s = 0.001, newx = as.matrix(x_train_df))
eval_results(y_train, predictions_train, x_train_df) #RMSE 0.51  R^2=0.72

# Prediction and evaluation on test data
predictions_test <- predict(ridge_reg, s = 0.001, newx = as.matrix(x_test_df))
eval_results(y_test, predictions_test, x_test_df) #RMSE 0.53  R^2=0.74

  #Metrics

rmse(y_test, predictions_test) #0.53
Metrics::mse(y_test, predictions_test) #0.29
Metrics::mae(y_test, predictions_test) #0.366
Metrics::rse(y_test, predictions_test) #0.25


x = 1:length(y_test)

plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, predictions_test, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()


# Lasso Regression


lambdas <- 10^seq(2, -3, by = -.1)

# Setting alpha = 1 implements lasso regression
lasso_reg <- cv.glmnet(as.matrix(x_train_df), y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)


# Best
lambda_best <- lasso_reg$lambda.min
lambda_best


lasso_model <- glmnet(as.matrix(x_train_df), y_train, alpha = 1, lambda = lambda_best, standardize = TRUE)


predictions_train <- predict(lasso_model, s = lambda_best, newx = as.matrix(x_train_df))
eval_results(y_train, predictions_train, x_train_df) #RMSE 0.51 #R^2 0.72

predictions_test <- predict(lasso_model, s = lambda_best, newx = as.matrix(x_test_df))
eval_results(y_test, predictions_test, x_test_df) #RMSE 0.54 #R^2 0.74

#Metrics

rmse(y_test, predictions_test) #0.541
Metrics::mse(y_test, predictions_test) #0.293
Metrics::mae(y_test, predictions_test) #0.367
Metrics::rse(y_test, predictions_test) #0.311


plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, predictions_test, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()



#Elastic Net Regression
library(plyr)
library(readr)
library(dplyr)
library(caret)
library(ggplot2)
library(repr)



#make train and test df

smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

x_train_df <- data.frame(df[train_ind, ])
y_train<-x_train_df$RefSt
#x_train_df = subset(x_train_df, select = - RefSt)

x_test_df <- data.frame(df[-train_ind, ])
y_test<-x_test_df$RefSt
#x_test_df = subset(x_test_df, select = - RefSt )




# Set training control



train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)



# Train the model
elastic_reg <- train( RefSt~ .,
                      data = x_train_df,
                      method = "glmnet",
                      preProcess = c("center", "scale"),
                      tuneLength = 10,
                      trControl = train_cont)



# Best tuning parameter
#CHECK if it is already in train method inside
elastic_reg$bestTune

# Make predictions on training set
predictions_train <- predict(elastic_reg, x_train_df)
eval_results(y_train, predictions_train, x_train_df) #RMSE 0.51 #R^2 0.73


# Make predictions on test set
predictions_test <- predict(elastic_reg, x_test_df)
eval_results(y_test, predictions_test, x_test_df) ##RMSE 0.54 #R^2 0.72


rmse(y_test, predictions_test) #0.49
Metrics::mse(y_test, predictions_test) #0.24
Metrics::mae(y_test, predictions_test) #0.352
Metrics::rse(y_test, predictions_test) #0.271


plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, predictions_test, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()


# SVM Regression https://koalatea.io/r-svm-regression/
library(e1071)
library(kernlab)
#Basic Model on the whole training set
model <- train(
  RefSt ~ .,
  data = df,
  method = 'svmRadial'
)
model

#Preprocessing with Caret

model2 <- train(
  RefSt ~ .,
  data = df,
  method = 'svmRadial',
  preProcess = c("center", "scale")
)
model2


model3 <- train(
  RefSt ~ .,
  data = x_train_df,
  method = 'svmRadial',
  preProcess = c("center", "scale")
)
model3

test.features = subset(x_test_df, select=-c(RefSt))
test.target = subset(x_test_df, select=RefSt)[,1]

predictions = predict(model3, newdata = test.features)

# RMSE 0.39
sqrt(mean((test.target - predictions)^2))

# R2 0.823
cor(test.target, predictions) ^ 2


#Cross Validation

ctrl <- trainControl(
  method = "cv",
  number = 10,
)

model4 <- train(
  RefSt ~ .,
  data = testing,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trCtrl = ctrl
)
model4


test.features = subset(x_test_df, select=-c(RefSt))
test.target = subset(x_test_df, select=RefSt)[,1]

predictions = predict(model4, newdata = test.features)

# RMSE
sqrt(mean((test.target - predictions)^2))

#  R2 0.838
cor(test.target, predictions) ^ 2


# Tuning Hyper Parameter
tuneGrid <- expand.grid(
  C = c(0.25, .5, 1),
  sigma = 0.1
)

model5 <- train(
  RefSt ~ .,
  data = training,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = ctrl,
  tuneGrid = tuneGrid
)
model5

plot(model5)


rmse(y_test, predictions) #0.39
Metrics::mse(y_test, predictions) #0.155
Metrics::mae(y_test, predictions) #0.260
Metrics::rse(y_test, predictions) #0.176


plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, predictions, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()



# KNN Model
knnmodel = knnreg(x_train_df, y_train)

pred_y = predict(knnmodel, data.frame(x_test_df))

mse = mean((y_test - pred_y)^2)
mae = caret::MAE(y_test, pred_y)
rmse = caret::RMSE(y_test, pred_y)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse) #MSE:  0.01926945 MAE:  0.08361779  RMSE:  0.1388144

x = 1:length(y_test)

plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, pred_y, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()



# Regression Random Forrest
library(randomForest)

rf_model <- randomForest(RefSt ~ ., data=x_train_df, ntree=1000,mtry = 13,
                         keep.forest=TRUE, importance=TRUE)

rf_model

ImpData <- as.data.frame(importance(rf_model))
ImpData$Var.Names <- row.names(ImpData)

ggplot(ImpData, aes(x=Var.Names, y=`%IncMSE`)) +
  geom_segment( aes(x=Var.Names, xend=Var.Names, y=0, yend=`%IncMSE`), color="skyblue") +
  geom_point(aes(size = IncNodePurity), color="blue", alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    legend.position="bottom",
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )


# mtry=13 or 15 is with lowest OOB
model_tuned <- tuneRF(
  x=x_train_df, #define predictor variables
  y=y_train, #define response variable
  ntreeTry=500,
  mtryStart=4,
  stepFactor=1.5,
  improve=0.01,
  trace=FALSE #don't show real-time progress
)

head(y_train)

y_pred<-predict(rf_model, newdata = x_test_df)

y_pred

y_test


mse = mean((y_test - y_pred)^2)
mae = caret::MAE(y_test, y_pred)
rmse = caret::RMSE(y_test, y_pred)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse) #MSE:  0.1634886 MAE:  0.250071  RMSE:  0.4043372


rmse(y_test, y_pred) #0.4043372
Metrics::mse(y_test, y_pred) #0.1634886
Metrics::mae(y_test, y_pred) #0.250071
Metrics::rse(y_test, y_pred) #0.1847949




plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, y_pred, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()



# DEEP NETWORK and ENSEMBLE


#Fit Regression Data with CNN Model in R
library(keras)
library(caret)

dim(x_train_df)
dim(xtrain)
dim(y_train)
head(x_train_df)
dim(x_test_df)

xtrain = as.matrix(x_train_df[,-14])

ytrain = as.matrix(y_train)

xtest = as.matrix(x_test_df[,-14])

ytest = as.matrix(y_test)


head(xtrain)
head(ytrain)

#Next we reshape the x input data by adding another one-dimension
xtrain = array(xtrain, dim = c(nrow(xtrain), 14, 1))
xtest = array(xtest, dim = c(nrow(xtest), 14, 1))

in_dim = c(dim(xtrain)[2:3])
print(in_dim)

model = keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 2,
                input_shape = in_dim, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  loss = "mse",
  optimizer = "adam")

model %>% summary()


model %>% fit(xtrain, ytrain, epochs = 100, batch_size=16, verbose = 0)
scores = model %>% evaluate(xtrain, ytrain, verbose = 0)
print(scores)

ypred = model %>% predict(xtest)

cat("RMSE:", RMSE(ytest, ypred))
cat("MSE", mean((y_test-ypred)^2))
cat("MAE", MAE(ytest,ypred))


x_axes = seq(1:length(ypred))

par("mar")
par(mar=c(1,1,1,1))

plot(x_axes, y_test,
     col = "burlywood", type = "l", lwd = 2, ylab = "medv")

lines(x_axes, ypred, col = "red", type = "l", lwd = 2)
legend("topleft", legend = c("y-test", "y-pred"),
       col = c("burlywood", "red"), lty=1, cex=0.7, lwd=2, bty='n')



#


xtrain = as.matrix(x_train_df[,-14])

ytrain = as.matrix(y_train)

xtest = as.matrix(x_test_df[,-14])

ytest = as.matrix(y_test)


head(xtrain)
head(ytrain)

#Next we reshape the x input data by adding another one-dimension
xtrain = array(xtrain, dim = c(nrow(xtrain), 14, 1))
xtest = array(xtest, dim = c(nrow(xtest), 14, 1))

in_dim = c(dim(xtrain)[2:3])

model = keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 3,
                input_shape = in_dim, activation = "relu") %>%
  #layer_batch_normalization() %>%
  #layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%

  layer_dense(units = 1, activation = "linear")

model %>% compile(
  loss = "mse",
  optimizer = "adam")

model %>% summary()


model %>% fit(xtrain, ytrain, epochs = 500, batch_size=10, verbose = 0)
scores = model %>% evaluate(xtrain, ytrain, verbose = 0)
print(scores)


ypred = model %>% predict(xtest)


kak<-cbind(ypred,ytest)



x_axes = seq(1:length(ytest))

plot(x_axes, ytest,
     col = "burlywood", type = "l", lwd = 2, ylab = "medv")

lines(x_axes, matrix(ypred), col = "red", type = "l", lwd = 2)
legend("topleft", legend = c("y-test", "y-pred"),
       col = c("burlywood", "red"), lty=1, cex=0.7, lwd=2, bty='n')



cat("RMSE:", RMSE(ytest, ypred)) #0.44
cat("MSE", mean((y_test-ypred)^2)) #0.2
cat("MAE", MAE(ytest,ypred)) #0.28


#layer dropout


xtrain = as.matrix(x_train_df[,-14])

ytrain = as.matrix(y_train)

xtest = as.matrix(x_test_df[,-14])

ytest = as.matrix(y_test)


head(xtrain)
head(ytrain)

#Next we reshape the x input data by adding another one-dimension
xtrain = array(xtrain, dim = c(nrow(xtrain), 14, 1))
xtest = array(xtest, dim = c(nrow(xtest), 14, 1))

in_dim = c(dim(xtrain)[2:3])


model = keras_model_sequential() %>%
  layer_conv_1d(filters = 64, kernel_size = 3,
                input_shape = in_dim, activation = "relu") %>%
  #layer_batch_normalization() %>%
  #layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_flatten() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout (rate = 0.3) %>%
  layer_dense(units = 1, activation = "linear")

model %>% compile(
  loss = "mse",
  optimizer = "adam")

model %>% summary()


model %>% fit(xtrain, ytrain, epochs = 500, batch_size=10, verbose = 0)
scores = model %>% evaluate(xtrain, ytrain, verbose = 0)
print(scores)


ypred = model %>% predict(xtest)


cat("RMSE:", RMSE(ytest, ypred)) #0.39
cat("MSE", mean((y_test-ypred)^2)) #0.15
cat("MAE", MAE(ytest,ypred)) #0.25



x_axes = seq(1:length(ytest))

plot(x_axes, ytest,
     col = "burlywood", type = "l", lwd = 2, ylab = "Sensor")

lines(x_axes, matrix(ypred), col = "red", type = "l", lwd = 2)
legend("topleft", legend = c("y-test", "y-pred"),
       col = c("burlywood", "red"), lty=1, cex=0.7, lwd=2, bty='n')
grid()



# ENSEMBLE METHOD

#caretEnsemble is a package for making ensembles of caret models
#caret ensemble has 3 primary functions like caretList, caretEnsemble and caretStack.
#CaretList is used to build lists of caret models od the same traning data with the same re-sampling parameters.
#CaretEnsemle and caretStack are used to create ensemlbe models from such lists of caret models.  (caretEnsemle uses a glm to create a simple linear blend of models
# and caretStack uses a caret model to combine the outputs from several component caret models.

#https://www.pluralsight.com/guides/ensemble-modeling-with-r
#https://cran.r-project.org/web/packages/caretEnsemble/vignettes/caretEnsemble-intro.html
#https://topepo.github.io/caret/available-models.html
#


install.packages("caretEnsemble")
library(caretEnsemble)


library("caret")
library("mlbench")
library("pROC")

#Data
df<-read.csv("C:\\FHWn\\Master-AGES\\8_Data\\complete_data.csv")
head(df)

#take away the python rownumber
df<-subset(df, select=-c(X))

#Scale DF
df<-scale(df, TRUE, TRUE)
summary(df)



#make train and test df

#data(Sonar)
smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

x_train_df <- data.frame(df[train_ind, ])
y_train<-x_train_df$RefSt
#x_train_df = subset(x_train_df, select = - RefSt)

x_test_df <- data.frame(df[-train_ind, ])
y_test<-x_test_df$RefSt
#x_test_df = subset(x_test_df, select = - RefSt )



my_control <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=FALSE,
  index=createResample(x_train_df$RefSt, 25),

)

library("rpart")
library("caretEnsemble")


model_list <- caretList(
  RefSt~., data=x_train_df,
  trControl=my_control,
  methodList=c('rpart', 'glm', 'knn', 'svmRadial')
)


p <- as.data.frame(predict(model_list, newdata=head(x_test_df)))
print(p)

#big model list

library("mlbench")
library("randomForest")
library("nnet")


#model_list_big <- caretList(
# RefSt~., data=x_train_df,
#trControl=my_control,
# metric="RMSE",
# methodList=c("glm", "rpart"),
# tuneList=list(
#   rf1=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=2)),
#   rf2=caretModelSpec(method="rf", tuneGrid=data.frame(.mtry=10), preProcess="pca"),
#   nn=caretModelSpec(method="nnet", tuneLength=2, trace=FALSE)
# )
#)


modelCor(resamples(model_list))


greedy_ensemble <- caretEnsemble(
  model_list,
  metric="RMSE",
  trControl=trainControl(
    number=2,

    classProbs=FALSE
  ))
summary(greedy_ensemble)

install.packages("caTools")
library("caTools")


glm_ensemble <- caretStack(
  model_list,
  method="glm",
  metric="RMSE",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=FALSE

  )
)
model_preds2 <- model_preds

model_preds2$ensemble <- predict(glm_ensemble, newdata=x_test_df)

CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
colAUC(model_preds2, x_test_df$RefSt)

#TRY TO SEE WHAT IS THE FIT
library("Metrics")

rmse(y_test, model_preds2$ensemble) #0.35
Metrics::mse(y_test, model_preds2$ensemble) #0.12
Metrics::mae(y_test, model_preds2$ensemble) #0.22
Metrics::rse(y_test, model_preds2$ensemble) #0.14



x = 1:length(y_test)
plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, model_preds2$ensemble, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()


# TO change for different types of models in ensemble!!!!
#***************************************************
#***************************************************
#c('rpart', 'glm', 'knn', 'svmRadial','rf','blasso')


my_control <- trainControl(
  method="boot",
  number=25,
  savePredictions="final",
  classProbs=FALSE,
  index=createResample(x_train_df$RefSt, 25),

)

library("rpart")
library("caretEnsemble")
library("randomForest")
library("monomvn")

model_list <- caretList(
  RefSt~., data=x_train_df,
  trControl=my_control,
  methodList=c('rpart', 'glm', 'knn', 'svmRadial','rf','blasso')
)


p <- as.data.frame(predict(model_list, newdata=head(x_test_df)))
print(p)

#big model list

library("mlbench")
library("randomForest")
library("nnet")




modelCor(resamples(model_list))


greedy_ensemble <- caretEnsemble(
  model_list,
  metric="RMSE",
  trControl=trainControl(
    number=2,

    classProbs=FALSE
  ))
summary(greedy_ensemble)

install.packages("caTools")
library("caTools")

library("kernlab")






#method of glm or to krlsPoly
glm_ensemble <- caretStack(

  model_list,
  method="rvmLinear",
  metric="RMSE",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=FALSE

  )
)
model_preds2 <- model_preds

model_preds2$ensemble <- predict(glm_ensemble, newdata=x_test_df)

CF <- coef(glm_ensemble$ens_model$finalModel)[-1]
colAUC(model_preds2, x_test_df$RefSt)

#TRY TO SEE WHAT IS THE FIT
library("Metrics")

rmse(y_test, model_preds2$ensemble) #0.35
Metrics::mse(y_test, model_preds2$ensemble) #0.12
Metrics::mae(y_test, model_preds2$ensemble) #0.22
Metrics::rse(y_test, model_preds2$ensemble) #0.14



x = 1:length(y_test)
plot(x, y_test, col = "red", type = "l", lwd=2,
     main = "ACTUAL vs PREDICTION")
lines(x, model_preds2$ensemble, col = "blue", lwd=2)
legend("topright",  legend = c("RefST", "Calibration"),
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()



# Elasticnet

install.packages("KRLS")
library("KRLS")


#Data
df<-read.csv("C:\\FHWn\\Master-AGES\\8_Data\\complete_data.csv")
head(df)

#take away the python rownumber
df<-subset(df, select=-c(X))

#Scale DF
df<-scale(df, TRUE, TRUE)
summary(df)



#make train and test df

#data(Sonar)
smp_size <- floor(0.75 * nrow(df))
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

x_train_df <- data.frame(df[train_ind, ])
y_train<-x_train_df$RefSt
x_train_df = subset(x_train_df, select = - RefSt)

x_test_df <- data.frame(df[-train_ind, ])
y_test<-x_test_df$RefSt
x_test_df = subset(x_test_df, select = - RefSt )

# Model KRLS fit
model_KRLS<-krls(x_train_df, y_train, whichkernel = "gaussian")









# Try rvmRadial, gaussprRadial, gaussprPoly
