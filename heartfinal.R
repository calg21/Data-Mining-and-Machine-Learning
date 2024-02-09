library(dplyr)
library(tidyverse)
library(caret)
library(psych)
library(caret)
library(e1071)

df22=read.csv('C:/Users/Calista/Desktop/DMML/DATASET/heart.csv')
df22 <- df22 %>%select(-State, -RemovedTeeth, -SmokerStatus, -RaceEthnicityCategory)

numeric_heart<-df22[sapply(df22, is.numeric)]
nrow(df22)


#check duplicate count
sum(duplicated(df22))
# remove duplicate


df22 <- df22 %>%
  distinct(across(everything()), .keep_all = TRUE)

nrow(df22) #124 duplicates removed

# Boxplots
par(mfrow=c(1,4))
boxplot(df22$WeightInKilograms)
title(main = "WeightInKilograms")
boxplot(df22$HeightInMeters)
title(main = "HeightInMeters")

# Boxplot for BMI
boxplot(df22$BMI)
title(main = "BMI")

# Boxplot for MentalHealthDays
boxplot(df22$MentalHealthDays)
title(main = "MentalHealthDays")
df22 <- df22[!(df22$WeightInKilograms > 250), ] #12outliers
nrow(df22)
df22 <- df22[!(df22$HeightInMeters <1.0), ] #10outliers
nrow(df22)
df22 <- df22[!(df22$BMI >90), ] #10outliers
nrow(df22)
##skewness
print(describe(numeric_heart))
df22$PhysicalHealthDays<-log(df22$PhysicalHealthDays+1)
df22$MentalHealthDays<-log(df22$MentalHealthDays+1)
numeric_heart<-df22[sapply(df22, is.numeric)]
print(describe(numeric_heart))

df22[] <- lapply(df22, function(x) if(is.character(x)) as.factor(x) else x)

#  categorical variables
categorical_vars <- df22 %>%
  select_if(is.factor)



chi_square_results <- list()


for (col in names(categorical_vars)) {

  contingency_table <- table(df22[[col]], df22$HadHeartAttack)
  

  chi_square_result <- chisq.test(contingency_table)
  

  chi_square_results[[col]] <- chi_square_result
}


for (i in seq_along(chi_square_results)) {
  cat("Chi-square test result for", names(chi_square_results)[i], ":\n")
  print(chi_square_results[[i]])
  cat("\n")
}



categorical_columns <- sapply(df22, is.factor)

# Convert  to factors
df22[categorical_columns] <- lapply(df22[categorical_columns], factor)

# label encoding
df22[categorical_columns] <- lapply(df22[categorical_columns], as.integer)

# null values
null_columns <- colSums(is.na(df22))

print(null_columns[null_columns > 0])

num_heart<-df22[sapply(df22, is.numeric)]
df22_numericcheck <- df22[, c('PhysicalHealthDays','MentalHealthDays','WeightInKilograms','BMI','HeightInMeters','SleepHours','HadHeartAttack')]
anova_result_numeric <- list()
for (variable in names(df22_numericcheck)[2:length(df22_numericcheck)]) {  # Exclude 'HadHeartAttack' from the loop
  
  # ANOVA
  anova_result <- aov(HadHeartAttack ~ get(variable), data = df22_numericcheck)
  
  
  anova_result_numeric[[variable]] <- summary(anova_result)
}

for (variable in names(anova_result_numeric)) {
  cat("ANOVA for", variable, "\n")
  print(anova_result_numeric[[variable]])
  cat("\n")
}
df22_numericcheck$PhysicalHealthDays <- as.numeric(factor(df22_numericcheck$PhysicalHealthDays))

anova_hadheart <- aov( HadHeartAttack~ PhysicalHealthDays, data =df22_numericcheck )


summary(anova_hadheart)


library(dplyr)
df22_subset <- df22[, c('HadHeartAttack', 'HadAngina', 'HadCOPD', 'AlcoholDrinkers', 'HadStroke', 
                      'HadDepressiveDisorder', 'DifficultyWalking', 'Sex', 
                      'HadDiabetes', 'PhysicalActivities', 'HadAsthma', 'DeafOrHardOfHearing', 
                      'BlindOrVisionDifficulty', 'DifficultyDressingBathing', 'PneumoVaxEver', 
                      'HighRiskLastYear', 'CovidPos', 'HadKidneyDisease', 
                      'HadSkinCancer', 'AgeCategory', 'GeneralHealth','PhysicalHealthDays','MentalHealthDays','WeightInKilograms','BMI','HeightInMeters','LastCheckupTime','ECigaretteUsage','ChestScan','TetanusLast10Tdap','HIVTesting',	'FluVaxLast12',
                      'DifficultyErrands','DifficultyConcentrating','HadArthritis')]



df22_subset$HadHeartAttack <- ifelse(df22_subset$HadHeartAttack == 1, 0, 1)
# Create a copy of df22
df3 <- df22_subset


y <- df3$HadHeartAttack
X <- df3

X<-  X %>%select(-HadHeartAttack)


set.seed(22186077)


indices <- createDataPartition(y=df3$HadHeartAttack, p = 0.7, list = FALSE)

# Split the data
X_train <- X[indices, ]
X_test <- X[-indices, ]
y_train <- y[indices]
y_test <- y[-indices]


# logistic regression model
model <- glm( y_train~ ., data = X_train, family = "binomial")
yy_train <- as.factor(y_train)

ctrl <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
# cross-validation using train
cv_model <- train(yy_train ~ ., data = cbind(X_train, yy_train), method = "glm", trControl = ctrl)

#cross-validation results
print(cv_model)


predictions <- predict(model, newdata = X_test, type = "response")


predicted_classes <- ifelse(predictions > 0.3, 1, 0)


predicted_classes <- factor(predicted_classes, levels = c(0, 1))


y_test <- factor(y_test, levels = c(0, 1))

conf_matrixlog <- confusionMatrix(predicted_classes, y_test)

print(conf_matrixlog)
library(pROC)

roc_curve <- roc(as.numeric((y_test)), as.numeric((predicted_classes)))
par(mfrow=c(1,1))
plot(roc_curve, col = "red", main = "ROC Curve", lwd = 2)
lines(x = c(0, 1), y = c(0, 1), col = "blue", lty = 2, lwd = 2)
legend("bottomright", legend = c("Logistic Regression"), col = c("red"), lty = 1, lwd = 2)
cat("AUC:", auc(roc_curve), "\n")

###########parameterized function  
library(caret)

evaluate_logistic_regression <- function(X_train, X_test, y_train, y_test, thresholds = c(0.3, 0.5)) {
  results <- data.frame()
  
  for (threshold in thresholds) {
    #logistic regression model
    model <- glm(y_train ~ ., data = X_train, family = "binomial")
    
    
  
    predictions <- predict(model, newdata = X_test, type = "response")
    
    
    predicted_classes <- ifelse(predictions > threshold, 1, 0)
    

    predicted_classes <- factor(predicted_classes, levels = c(0, 1))
    y_test <- factor(y_test, levels = c(0, 1))
    
    
    conf_matrix <- confusionMatrix(predicted_classes, y_test)
    
    
    cat("Threshold:", threshold, "\n")
    print(conf_matrix)
    cat("Accuracy: ", conf_matrix$overall["Accuracy"], "\n\n")
    
    # Store the results in a data frame
    results <- rbind(results, data.frame(
      Threshold = threshold,
      Accuracy = conf_matrix$overall["Accuracy"],
      Precision = conf_matrix$byClass["Pos Pred Value"],
      Recall = conf_matrix$byClass["Sensitivity"],
      F1_Score = 2 * (conf_matrix$byClass["Pos Pred Value"] * conf_matrix$byClass["Sensitivity"]) / 
        (conf_matrix$byClass["Pos Pred Value"] + conf_matrix$byClass["Sensitivity"])
    ))
  }
  
  return(results)
}

#  thresholds 0.3 and 0.5
result_df <- evaluate_logistic_regression(X_train, X_test, y_train, y_test, thresholds = c(0.3, 0.5))


print(result_df)



#####################EXTRA(Naive bayes)

#  Naive Bayes model
modelnaive <- naiveBayes(y_train ~ ., data =  X_train, weight = c(1, 10))


print(modelnaive)


predictionsn <- predict(modelnaive, newdata = X_test)


predictions <- factor(predictionsn, levels = levels(y_test))


conf_matrix <- table(predictions, y_test)
conf_matrix <-confusionMatrix(predictions, y_test)
print(conf_matrix)


accuracy <- conf_matrix$overall["Accuracy"]


precision <- conf_matrix$byClass["Pos Pred Value"]


recall <- conf_matrix$byClass["Sensitivity"]


f1_score <- 2 * (precision * recall) / (precision + recall)


cat("Accuracy: ", accuracy, "\n")
cat("Precision: ", precision, "\n")
cat("Recall: ", recall, "\n")
cat("F1 Score: ", f1_score, "\n")






