#read data & split data
```{r}
library(glmnet)
library(gbm)
library(caret)
library(tree)
library(pROC)
library(randomForest)
diabetes <- read.csv("C:/Users/minghong xie/Desktop/diabetes_data_upload.csv")
binary_features <- c("Gender","Polyuria", "Polydipsia", "sudden.weight.loss", "weakness", "Polyphagia", "Genital.thrush", "visual.blurring", "Itching", "Irritability", "delayed.healing", "partial.paresis", "muscle.stiffness", "Alopecia", "Obesity", "class")

diabetes[binary_features] <- lapply(diabetes[binary_features], factor)
set.seed(123)
index <- sample(nrow(diabetes), floor(0.8 * nrow(diabetes)))
train_data <- diabetes[index, ]
test_data <- diabetes[-index, ]

```

#knn model
```{r}
K <- 5
fold_ind <- sample(1:K, nrow(train_data), replace = TRUE)
K_seq <- seq(from = 1, to = 100, by = 1)
CV_error_seq <- sapply(K_seq, function(K_cur) {
  mean(sapply(1:K, function(j) {
    fit_knn <- knn3(class ~ ., data = train_data[fold_ind != j, ], k = K_cur)
    pred_knn <- predict(fit_knn, newdata = train_data[fold_ind == j, ], type = "class")
    mean(pred_knn != train_data$class[fold_ind == j])
  }))
})
knn_re <- data.frame(K = K_seq, CV_error = CV_error_seq)
best_k <- K_seq[which.min(knn_re$CV_error)]
knn_model <- knn3(class ~ ., data = train_data, k = 1)
knn_predictions<-predict(knn_model, newdata=test_data, type="class")
confusionMatrix(knn_predictions, test_data$class)


roc_obj <- roc(response = test_data$class, predictor = as.numeric(knn_predictions))
auc(roc_obj)
plot(roc_obj, main = "ROC Curve for KNN Model")

```

#decision tree
```{r}
dt_model<-tree(class ~ ., data = train_data)
cv.dt <- cv.tree(dt_model)
best_size <- cv.dt$size[which.min(cv.dt$dev)]
dia.tree <- prune.tree(dt_model, best = best_size)

plot(dia.tree)
text(dia.tree)
dt_predictions <- predict(dia.tree, newdata = test_data, type = "class")

confusionMatrix(dt_predictions, test_data$class)

roc_obj1 <- roc(response = test_data$class, predictor = as.numeric(dt_predictions))
auc(roc_obj1)
plot(roc_obj1, main = "ROC Curve for Decision Tree")
```
#rondom forest 
```{r}
rf_model <- randomForest(class ~ ., data = train_data, ntree=100)
print(rf_model)
varImpPlot(rf_model)
rf_predictions <- predict(rf_model, newdata = test_data)
confusionMatrix(rf_predictions, test_data$class)

roc_obj2 <- roc(response = test_data$class, predictor = as.numeric(rf_predictions))
auc(roc_obj2)
plot(roc_obj2, main = "ROC Curve for random forest")


```


#boosting
```{r}
train_data$class <- ifelse(train_data$class == "Positive", 1, 0)
test_data$class <- ifelse(test_data$class == "Positive", 1, 0)
boost_model<-gbm(class ~ ., data = train_data, distribution = "bernoulli", n.trees = 5000, 
                  interaction.depth = 4,shrinkage = 0.2, cv.folds = 5)
best_n_tress <- which.min(boost_model$cv.error)
summary(boost_model)

yprob.boost <- predict(boost_model, newdata = test_data, n.trees = best_n_tress, type = "response")
predicted_class <- ifelse(yprob.boost > 0.5, 1, 0)  # 确保结果为数字

actual <- factor(test_data$class, levels = c(0, 1))
predicted_class <- factor(predicted_class, levels = c(0, 1))

confusionMatrix(predicted_class, actual)

roc_obj3 <- roc(response = test_data$class, predictor = as.numeric(predicted_class))

auc(roc_obj3)

plot(roc_obj3, main = "ROC Curve for Boosting")
```

#LASSO 
```{r}
train_matrix <- model.matrix(~ . - class, data = train_data)
test_matrix<- model.matrix(~ . - class, data = test_data)
lasso_model <- cv.glmnet(
  x = train_matrix,
  y = train_data$class,
  family = "binomial",
  alpha = 1
)
best_lambda <- lasso_model$lambda.min
LASSO_predictions <- predict(
  lasso_model, 
  s = best_lambda, 
  newx = test_matrix, 
  type = "response"
)

LASSO_predictions <- ifelse(LASSO_predictions>0.5,1,0)
LASSO_predictions<-as.factor(LASSO_predictions)
test_data$class<-as.factor(test_data$class)
confusionMatrix(LASSO_predictions, test_data$class)

roc_obj4 <- roc(response = test_data$class, predictor = as.numeric(LASSO_predictions))
auc(roc_obj4)
plot(roc_obj4, main = "ROC Curve for LASSO")
```

