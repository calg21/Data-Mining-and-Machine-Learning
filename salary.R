library(dplyr)
library(randomForest)
library(psych)
library(caret)
salary=read.csv('C:/Users/Calista/Desktop/DMML/DATASET/salary.csv')
nrow(salary)
categorical_columns <- sapply(salary, function(x) is.character(x) || is.factor(x))
categorical_data <- salary[, categorical_columns]

# Display all unique categories for each categorical column
unique_categories <- lapply(categorical_data, function(x) unique(x))
print(unique_categories)

#find the column name and column type
summary_salary <- data.frame( Column_Type = sapply(salary, class))
print(summary_salary)
#check duplicates
sum(duplicated(salary))  #24 duplicates
salary <- salary %>%
  distinct(across(everything()), .keep_all = TRUE)

sum(duplicated(salary))
nrow(salary)
#check null values
null_columns <- colSums(is.na(salary))

print(null_columns[null_columns > 0])
#outliers

boxplot(salary$age)
salary <- salary[!(salary$age > 89), ] #42outliers
nrow(salary)
boxplot(salary$fnlwgt)
salary <- salary[!(salary$fnlwgt > 1000000), ] #13outliers
nrow(salary)
boxplot(salary$education.num)
boxplot(salary$capital.gain)
boxplot(salary$capital.loss)
salary <- salary[!(salary$capital.loss > 4000), ] #2outliers
nrow(salary)
#check skewness
numeric_cols <- salary[sapply(salary, is.numeric)]


stats <- describe(numeric_cols)

print(stats)
salary$fnlwgt<-log1p(salary$fnlwgt)
salary$capital.loss<-log1p(salary$capital.loss)
salary$capital.gain<-log1p(salary$capital.gain)

# Find columns containing '?'
col_qt_mark <- colnames(salary)[sapply(salary, function(col) any(grepl("\\?", col)))]

######converting ? to Not available
salary$native.country <- ifelse(trimws(salary$native.country) == '?', 'NotAvailable', salary$native.country)

salary$occupation<- ifelse(trimws(salary$occupation) == '?', 'NotAvailable', salary$occupation
)
salary$workclass<- ifelse(trimws(salary$workclass) == '?', 'NotAvailable', salary$workclass
)
#apply chi square
chi_square_results <- list()

categorical_columns <- sapply(salary, is.character)
character_columns <- names(categorical_columns)[categorical_columns]

# Iterate through each character variable
for (col in character_columns) {
  
  contingency_table <- table(salary[[col]], salary$salary)
  
 
  chi_square_result <- chisq.test(contingency_table)
  
  
  chi_square_results[[col]] <- chi_square_result
}


for (i in seq_along(chi_square_results)) {
  cat("Chi-square test result for", names(chi_square_results)[i], ":\n")
  print(chi_square_results[[i]])
  cat("\n")
}


salary$salary <- as.numeric(as.factor(salary$salary))

salary$age <- as.numeric(factor(salary$age))

anova_age <- aov(salary ~ age, data = salary)

#anova to check for every numeric variable
summary(anova_age)
salary$capital.gain <- as.numeric(factor(salary$capital.gain))

anova_capital.gain <- aov(salary ~ capital.gain, data = salary)


summary(anova_capital.gain)


salary$capital.loss <- as.numeric(factor(salary$capital.loss))

anova_capital.loss <- aov(salary ~ capital.loss, data = salary)


summary(anova_capital.loss)


salary$hours.per.week <- as.numeric(factor(salary$hours.per.week))

anova_hours.per.week <- aov(salary ~ hours.per.week, data = salary)


summary(anova_hours.per.week)

salary$education.num<- as.numeric(factor(salary$education.num
))

anova_education.num<- aov(salary ~ education.num
                          , data = salary)


summary(anova_education.num)

#grouping all categories of one type
salary <- salary %>%
  mutate(
    workclass = case_when(
      grepl("Self-emp", workclass) ~ "Self Employed",
      grepl("Without-pay|Never-worked", workclass, ignore.case = TRUE) ~ "Unemployed",
      grepl("State-gov|Federal-gov|Local-gov", workclass, ignore.case = TRUE) ~ "Govt.",
      TRUE ~ workclass
    )
  )



salary <- salary %>%
  mutate(
    marital.status = case_when(
      grepl("Married-civ-spouse|Married-AF-spouse", marital.status) ~ "Married",
      grepl("Divorced|Married-spouse-absent|Separated|Widowed", marital.status) ~ "Single",
      TRUE ~ marital.status
    )
  )


salary <- salary %>%
  mutate(
    native.country = case_when(
      grepl("United-States", native.country) ~ 'USA',
      grepl("South", native.country) ~ 'SouthKorea',
      grepl("Puerto-Rico", native.country) ~ 'PuertoRico',
      grepl("Dominican-Republic", native.country) ~ 'DominicRep',
      grepl("Outlying-US(Guam-USVI-etc)", native.country) ~ 'OutlyingUSA',
      grepl("Trinadad&Tobago", native.country) ~ 'Tri&Tob',
      grepl("Holand-Netherlands", native.country) ~ 'Netherlands',
      grepl("Hong", native.country) ~ 'HongKong',
      TRUE ~ native.country
    )
  )




numeric_columns <- names(salary)[sapply(salary, is.numeric)]

# label encoding
categorical_columns <- sapply(salary, is.character)

salary[categorical_columns] <- lapply(salary[categorical_columns], factor)

salary[categorical_columns] <- lapply(salary[categorical_columns], as.integer)

#check null values
null_columns <- colSums(is.na(salary))

print(null_columns[null_columns > 0])

#Random Forest model
rf_model <- randomForest(salary ~ ., data = salary, importance = TRUE, ntree = 100)


print(rf_model$importance)


varImpPlot(rf_model)

#drop columns with less importance
salary<-salary%>%select(-race,-sex,-native.country)
salary$salary <- ifelse(salary$salary == 1, 0, 1)
y <-salary$salary

x <- salary
set.seed(22186077)
x<-x %>%select(-salary)
splitIndex <- createDataPartition(y = salary$salary, p = 0.7, list = FALSE)
x_train_salary <- x[splitIndex, ]
x_test_salary <- x[-splitIndex, ]
y_train_salary <- y[splitIndex]
y_test_salary <- y[-splitIndex]


#  Naive Bayes model(EXTRA MODEL)
library(e1071)

nb_model <- naiveBayes(y_train_salary ~ ., data = x_train_salary)

#predictions 
y_pred <- predict(nb_model, newdata = x_test_salary)

y_pred <- factor(y_pred, levels = c(0, 1))
y_test_salary <- factor(y_test_salary, levels = c(0, 1))


conf_matrix <- confusionMatrix(y_pred, y_test_salary)


print(conf_matrix)






##############################KNN

library(class)
library(class)
library(caret)

# the range of k values to consider
k_values <- seq(1, 10, by = 2)  

# store cross-validation results
cv_results <- data.frame(k = numeric(0), Accuracy = numeric(0))


set.seed(123)  
for (k in k_values) {
  
  knn_model <- knn(train = x_train_salary, test = x_test_salary, cl = y_train_salary, k = k)
  

  y_pred <- as.factor(knn_model)
  
  
  accuracy <- mean(y_pred == y_test_salary)
  
  
  cv_results <- rbind(cv_results, data.frame(k = k, Accuracy = accuracy))
}


print(cv_results)

# Identify the best k based on the highest accuracy
best_k <- cv_results$k[which.max(cv_results$Accuracy)]
cat("Best k:", best_k, "\n")

# Train KNN model
knn_model <- knn(train = x_train_salary, test = x_test_salary, cl = y_train_salary, k =best_k )

# predictions
y_pred_knn <- as.factor(knn_model)
y_test_salary <- factor(y_test_salary, levels = c(0, 1))
# Evaluate the model
conf_matrix_knn <- confusionMatrix(y_test_salary, y_pred_knn)
print(conf_matrix_knn)

library(pROC)

roc_curve <- roc(as.numeric((y_test_salary)), as.numeric((y_pred_knn)))

plot(roc_curve, col = "red", main = "ROC Curve", lwd = 2)
lines(x = c(0, 1), y = c(0, 1), col = "blue", lty = 2, lwd = 2)
legend("bottomright", legend = c("KNN"), col = c("red"), lty = 1, lwd = 2)
cat("AUC:", auc(roc_curve), "\n")