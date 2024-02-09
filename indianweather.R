library(caret)
library(dplyr)
library(psych)
library(ggplot2)
library(corrplot)
library(randomForest)
library(glmnet)
library(gridExtra)
weather=read.csv('C:/Users/Calista/Downloads/DATASET/IndianWeatherRepository.csv')
#find the column name and column type
summary_weather <- data.frame( Column_Type = sapply(weather, class))
print(summary_weather)

null_values <- is.na(weather)
colnames(weather)
sum(null_values)

#checking if any duplicate values are present or not
sum(duplicated(weather))
nrow(weather)

#dropping unnecessary columns
weather <- weather %>%select(-temperature_fahrenheit, -wind_mph, -pressure_in, -precip_mm, -feels_like_fahrenheit, -visibility_km, -gust_mph)

#Random forest variable importance
rf_modelweather <- randomForest( temperature_celsius~ ., data = weather, importance = TRUE, ntree = 100)

print(rf_modelweather$importance)


varImpPlot(rf_modelweather)






#ANOVA TESTING
weather$condition_text <- as.numeric(factor(weather$condition_text))
anova_temp <- aov(temperature_celsius ~ weather$condition_text, data = weather)
summary(anova_temp)


weather$region <- as.numeric(factor(weather$region))
anova_region <- aov(temperature_celsius ~ weather$region, data = weather)
summary(anova_region)

#Correlation
numericweather<- weather[sapply(weather, is.numeric)]
co<- cor(weather$temperature_celsius, numericweather)

a<-ggplot(data=weather,aes(y=temperature_celsius
                    ,x=pressure_mb))+ geom_point()
b<-ggplot(data=weather,aes(y=temperature_celsius
                        ,x=wind_kph))+ geom_point()
c<-ggplot(data=weather,aes(y=temperature_celsius
                        ,x=latitude))+ geom_point()
d<-ggplot(data=weather,aes(y=temperature_celsius
                        ,x=feels_like_celsius))+ geom_point()
e<-ggplot(data=weather,aes(y=temperature_celsius
                        ,x=uv_index))+ geom_point()
f<-ggplot(data=weather,aes(y=temperature_celsius
                        ,x=gust_kph))+ geom_point()
grid.arrange(a,b,c,d,e,f, ncol = 6)

#remove outliers
weather <- weather[!((weather$pressure_mb >=1025) &((weather$temperature_celsius)<10)), ] #3outliers
nrow(weather)

weather <- weather[!(weather$wind_kph > 38 & weather$temperature_celsius > 20), ] #6outliers
nrow(weather)
weather <- weather[!(weather$wind_kph < 35 & weather$temperature_celsius > 35), ] #9outliers
nrow(weather)
weather <- weather[!(weather$wind_kph >12 & weather$temperature_celsius < 5), ] #4outliers
nrow(weather)
weather <- weather[!(weather$feels_like_celsius >45 & weather$temperature_celsius > 30), ] #2outliers
nrow(weather)
weather <- weather[!(weather$uv_index >8 & weather$temperature_celsius > 30), ] #1outliers
nrow(weather)

weather <- weather[!(weather$air_quality_Carbon_Monoxide >7500 & weather$temperature_celsius > 20), ] #14outliers
nrow(weather)

weather <- weather[!(weather$gust_kph >60 & weather$temperature_celsius > 20), ] #6outliers
nrow(weather)

#check corelation after removing outliers
numericweather<- weather[sapply(weather, is.numeric)]
co<- cor(weather$temperature_celsius, numericweather)


temperature_data <- weather$temperature_celsius

# considering columns whose cor is  greater than +/-0.15
temperature_factors <- weather[, c('latitude','last_updated_epoch', 'wind_kph', 'pressure_mb','feels_like_celsius','uv_index','air_quality_Carbon_Monoxide'  ,'gust_kph','condition_text','region')]


#check for skewness
num_weather<-temperature_factors[sapply(temperature_factors, is.numeric)]
descriptive_weather <- describe(num_weather)
print(descriptive_weather)

#to handle skewness whose value is greater than 2.5
temperature_factors$air_quality_Carbon_Monoxide <- log1p(temperature_factors$air_quality_Carbon_Monoxide) 
temperature_factors$uv_index <- log1p(temperature_factors$uv_index)  

set.seed(22186077)

# Perform the train-test split
splitIndex <- createDataPartition(y = temperature_data, p = 0.7, list = FALSE)
x_train_weather <- temperature_factors[splitIndex, ]
x_test_weather <- temperature_factors[-splitIndex, ]
y_train_weather <- temperature_data[splitIndex]
y_test_weather <- temperature_data[-splitIndex]

cat("x_train_weather:", dim(x_train_weather), "\n")
cat("x_test_weather:", dim(x_test_weather), "\n")
cat("y_train_weather:", length(y_train_weather), "\n")
cat("y_test_weather:", length(y_test_weather), "\n")





#################linear model
lm_model <- lm(y_train_weather ~ ., data = x_train_weather)


y_predlin <- predict(lm_model, newdata = x_test_weather)

# Calculate R-squared
rsquaredlin <- 1 - sum((y_test_weather - y_predlin)^2) / sum((y_test_weather - mean(y_test_weather))^2)
cat("Accuracy of the model is", sprintf("%.2f", rsquaredlin))
mse <- mean((y_predlin - y_test_weather)^2)
rmse <- sqrt(mse)
cat("Mean Squared Error (MSE):", format(mse, digits = 2), "\n")
cat("Root Mean Squared Error (RMSE):", format(rmse, digits = 2), "\n")
mae <- mean(abs(y_predlin - y_test_weather))
cat("Mean Absolute Error (MAE):", format(mae, digits = 2), "\n")
# Create QQ plot
qqplot(y_test_weather, y_predlin, main = "QQ Plot", xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
abline(0, 1, col = "red")  # Add a line for perfect agreement




#############randomforest
library(randomForest)
#grid of hyperparameter values to search
param_grid <- expand.grid(
  ntree = c(50, 100),
  mtry = c(2, 4),
  max_depth = c(5, 10)
)
results <- list()

# Perform grid search
for (i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]
  
  # Train model with current hyperparameter values
  model <- randomForest(y_train_weather ~ ., data = x_train_weather, 
                        ntree = params$ntree, mtry = params$mtry, max_depth = params$max_depth)
  
  #predictions
  predictions <- predict(model, newdata = x_test_weather)
  
  #performance 
  mse <- mean((predictions - y_test_weather)^2)
  
  # Store results
  results[[paste("Model_", i)]] <- list(parameters = params, mse = mse)
}

# Find the best set based on lowest MSE
best_model <- results[[which.min(sapply(results, function(result) result$mse))]]
print(best_model$parameters)

#random forest regression model
rf_model <- randomForest(y_train_weather ~ ., data = x_train_weather, ntree = 100, mtry=4,max_depth=5)
 
#predictions
y_predrf <- predict(rf_model, newdata = x_test_weather)


rsquaredrf <- 1 - sum((y_test_weather - y_predrf)^2) / sum((y_test_weather - mean(y_test_weather))^2)
cat("Accuracy of the Random Forest model is", sprintf("%.2f", rsquaredrf))
mse <- mean((y_predrf - y_test_weather)^2)
rmse <- sqrt(mse)

cat("Mean Squared Error (MSE):", format(mse, digits = 2), "\n")
cat("Root Mean Squared Error (RMSE):", format(rmse, digits = 2), "\n")
mae <- mean(abs(y_predrf - y_test_weather))
cat("Mean Absolute Error (MAE):", format(mae, digits = 2), "\n")


qqplot(y_predrf,y_test_weather, main = "QQ Plot", xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
abline(0, 1, col = "red") 

###########################gradient boosting regression model
library(gbm)
train_data <- data.frame(y = y_train_weather, x_train_weather)


model <- train(
  y ~ .,
  data = train_data,
  method = "gbm",
  trControl = trainControl(
    method = "cv",
    number = 5,
    verboseIter = TRUE
  ),
  tuneGrid = expand.grid(
    n.trees = c(50, 100, 200),  
    interaction.depth = c(3, 5),  
    shrinkage = c(0.1, 0.01),        
    n.minobsinnode = c(5, 10, 15)    
  )
)

#optimal values based on cross-validation
optimal_params <- model$bestTune
print(optimal_params)
optimal_n_trees<-optimal_params$n.trees
#gradient boosting regression model
gbm_model <- gbm(y_train_weather ~ ., data = x_train_weather, n.trees = optimal_n_trees, shrinkage = 0.1, distribution = "gaussian", interaction.depth = 5, bag.fraction = 1, train.fraction = 1, n.minobsinnode = 10, verbose = TRUE, n.cores = 1)

#predictions on the test set
y_predgbm <- predict(gbm_model, newdata = x_test_weather, n.trees = optimal_n_trees)


rsquaredgbm <- 1 - sum((y_test_weather - y_predgbm)^2) / sum((y_test_weather - mean(y_test_weather))^2)
cat("Accuracy of the Gradient Boosting model is", sprintf("%.2f", rsquaredgbm))
mse <- mean((y_predgbm - y_test_weather)^2)


rmse <- sqrt(mse)

cat("Mean Squared Error (MSE):", format(mse, digits = 2), "\n")
cat("Root Mean Squared Error (RMSE):", format(rmse, digits = 2), "\n")
mae <- mean(abs(y_predgbm - y_test_weather))

cat("Mean Absolute Error (MAE):", format(mae, digits = 2), "\n")
qqplot(y_predgbm,y_test_weather, main = "QQ Plot", xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
abline(0, 1, col = "red") 
###################SVM

library(e1071)

#SVM regression model
svm_model <- svm(y_train_weather ~ ., data = x_train_weather)

#predictions
y_predsvm <- predict(svm_model, newdata = x_test_weather)





rsquaredsvm <- 1 - sum((y_test_weather - y_predsvm)^2) / sum((y_test_weather - mean(y_test_weather))^2)

cat("R-squared of the SVM model is", sprintf("%.2f", rsquaredsvm ), "\n")
mse <- mean((y_test_weather - y_predsvm)^2)
cat("Mean Squared Error (MSE) of the SVM model is", mse, "\n")


rmse <- sqrt(mse)
cat("Root Mean Squared Error (RMSE) of the SVM model is", rmse, "\n")


mae <- mean(abs(y_test_weather - y_predsvm))
cat("Mean Absolute Error (MAE) of the SVM model is", mae, "\n")

par(mfrow=c(1,1))
qqplot( y_predsvm,y_test_weather, main = "QQ Plot", xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
abline(0, 1, col = "red")  






