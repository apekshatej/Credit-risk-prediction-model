# classification package
install.packages("class")

# decision tree package
install.packages("party")

# this one has an odd name but we do need it - SVM package
install.packages("e1071")

# this is the crossvalidation package - CARET
install.packages("caret")

# now reference all the libraries
library(class)
library(party)
library(e1071)
library(caret)
library(ggplot2)
library(mfx)
library(randomForest)

# type CTRL+L to clean up your console

# cleanup your environment
rm(list=ls())

# set the working directory
setwd("~/Desktop/Uni_Applications/UW MISM/IMT 572-Intro to DS/Rfiles")

credit_dataset_CSV <- read.csv("C:/Users/tejwa/Desktop/Uni_Applications/UW MISM/IMT 572-Intro to DS/Rfiles/credit_dataset_CSV.csv", 
                               header = TRUE, stringsAsFactors = TRUE)


# the ID is one per row - not useful as a predictor - remove
credit_dataset_CSV$ID = NULL

# some variables are clearly representing categories --> as.factor() to convert
credit_dataset_CSV$EDUCATION = as.factor(credit_dataset_CSV$EDUCATION)
credit_dataset_CSV$SEX = as.factor(credit_dataset_CSV$SEX)
credit_dataset_CSV$MARRIAGE = as.factor(credit_dataset_CSV$MARRIAGE)
credit_dataset_CSV$PAY_0 = as.factor(credit_dataset_CSV$PAY_0)
credit_dataset_CSV$PAY_2 = as.factor(credit_dataset_CSV$PAY_2)
credit_dataset_CSV$PAY_3 = as.factor(credit_dataset_CSV$PAY_3)
credit_dataset_CSV$PAY_4 = as.factor(credit_dataset_CSV$PAY_4)
credit_dataset_CSV$PAY_5 = as.factor(credit_dataset_CSV$PAY_5)
credit_dataset_CSV$PAY_6 = as.factor(credit_dataset_CSV$PAY_6)

# summary statistics
summary(credit_dataset_CSV)

# check for missing values 
sum(is.na(credit_dataset_CSV))

# Plot histogram of AGE
hist(credit_dataset_CSV$AGE, main = "Histogram of AGE", xlab = "AGE", col = "green", border = "black")
# Visualise LIMIT_BAL as a histogram
hist(credit_dataset_CSV$LIMIT_BAL, main = "Histogram of LIMIT_BAL", xlab = "LIMIT_BAL", col = "purple", border = "black")

# Visualise BILL_AMT1 as a histogram
hist(credit_dataset_CSV$BILL_AMT1, main = "Histogram of BILL_AMT1", xlab = "BILL_AMT1", col = "orange", border = "black")

# Visualise PAY_AMT1 as a histogram
hist(credit_dataset_CSV$PAY_AMT1, main = "Histogram of PAY_AMT1", xlab = "PAY_AMT1", col = "red", border = "black")


# Normalise continuous variables
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

credit_dataset_CSV$LIMIT_BAL <- normalize(credit_dataset_CSV$LIMIT_BAL)
credit_dataset_CSV$AGE <- normalize(credit_dataset_CSV$AGE)
credit_dataset_CSV$BILL_AMT1 <- normalize(credit_dataset_CSV$BILL_AMT1)
credit_dataset_CSV$PAY_AMT1 <- normalize(credit_dataset_CSV$PAY_AMT1)


# Verify transformations
summary_transformed <- summary(credit_dataset_CSV)
print("Summary Statistics After Transformation:")
print(summary_transformed)

# Load necessary libraries
library(ggplot2)

# Function to check normalisation visually
plot_normalisation <- function(data, variable_name) {
  ggplot(data, aes_string(x = variable_name)) +
    geom_histogram(bins = 30, fill = "blue", colour = "black", alpha = 0.7) +
    theme_minimal() +
    ggtitle(paste("Distribution of", variable_name)) +
    xlab(variable_name) +
    ylab("Frequency")
}

# Plot the normalised variables
plot_list <- list(
  plot_normalisation(credit_dataset_CSV, "LIMIT_BAL"),
  plot_normalisation(credit_dataset_CSV, "AGE"),
  plot_normalisation(credit_dataset_CSV, "BILL_AMT1"),
  plot_normalisation(credit_dataset_CSV, "PAY_AMT1")
)

# Print plots
print(plot_list[[1]])
print(plot_list[[2]])
print(plot_list[[3]])
print(plot_list[[4]])

# Ensure the target variable is a factor and not included in the correlation matrix
credit_dataset_CSV$default.payment.next.month <- as.factor(credit_dataset_CSV$default.payment.next.month)

# Select only numeric independent variables
independent_numeric_vars <- credit_dataset_CSV[,sapply(credit_dataset_CSV, is.numeric) ]

# Calculate the correlation matrix for numeric independent variables
correlation_matrix <- cor(independent_numeric_vars, use = "pairwise.complete.obs")

# Print the correlation matrix
print(correlation_matrix)

# Visualise the correlation matrix
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", 
         tl.col = "black", tl.cex = 0.8, title = "Correlation Matrix")


#Predictor set 1
# Define the formula with selected variables
formula <- default.payment.next.month ~ LIMIT_BAL + PAY_AMT1 + PAY_AMT3 + BILL_AMT1 + PAY_0 + PAY_2 + PAY_3 + SEX + EDUCATION + MARRIAGE

# Run the Logit model
logit_model <- glm(formula, family = "binomial", data = credit_dataset_CSV)
summary(logit_model)

# Interpret marginal effects for Logit model
logit_marginal <- logitmfx(formula, data = credit_dataset_CSV)
print("Marginal Effects - Logit Model:")
print(logit_marginal)

# Run the Probit model
probit_model <- glm(formula, family = binomial(link = "probit"), data = credit_dataset_CSV)
summary(probit_model)

# Interpret marginal effects for Probit model
probit_marginal <- probitmfx(formula, data = credit_dataset_CSV)
print("Marginal Effects - Probit Model:")
print(probit_marginal)

# Compare the models using AIC
logit_aic <- AIC(logit_model)
probit_aic <- AIC(probit_model)

comparison <- data.frame(Model = c("Logit", "Probit"),
                         AIC = c(logit_aic, probit_aic))
print("Model Comparison (AIC):")
print(comparison)

#Predictor set 2
# Define the formula 2
formula_2 <- default.payment.next.month ~ LIMIT_BAL + PAY_AMT1 + PAY_AMT3 + PAY_0 + PAY_2 + PAY_3

# Run the Logit model
logit_model_2 <- glm(formula_2, family = "binomial", data = credit_dataset_CSV)
summary(logit_model_2)

# Run the Probit model
probit_model_2 <- glm(formula_2, family = binomial(link = "probit"), data = credit_dataset_CSV)
summary(probit_model_2)

# Compare AIC for Predictor Set 2
logit_aic_2 <- AIC(logit_model_2)
probit_aic_2 <- AIC(probit_model_2)
cat("Predictor Set 2 - AIC Comparison:\n")
cat("Logit AIC:", logit_aic_2, "\n")
cat("Probit AIC:", probit_aic_2, "\n")


#Predictor set 3
# Define the formula 3
formula_3 <- default.payment.next.month ~ LIMIT_BAL + BILL_AMT1 + PAY_0 + PAY_2 + SEX + EDUCATION + MARRIAGE

# Run the Logit model
logit_model_3 <- glm(formula_3, family = "binomial", data = credit_dataset_CSV)
summary(logit_model_3)

# Run the Probit model
probit_model_3 <- glm(formula_3, family = binomial(link = "probit"), data = credit_dataset_CSV)
summary(probit_model_3)

# Compare AIC for Predictor Set 3
logit_aic_3 <- AIC(logit_model_3)
probit_aic_3 <- AIC(probit_model_3)
cat("Predictor Set 3 - AIC Comparison:\n")
cat("Logit AIC:", logit_aic_3, "\n")
cat("Probit AIC:", probit_aic_3, "\n")

#Training a classifier on outcome default.

credit_dataset_CSV$default.payment.next.month = as.factor(credit_dataset_CSV$default.payment.next.month)
library("caret")
train_Control = trainControl(method = "cv", number = 5) 
knn_caret = train(default.payment.next.month~LIMIT_BAL + PAY_AMT1 + PAY_AMT3 + BILL_AMT1 + PAY_0 + PAY_2 + PAY_3 + SEX + EDUCATION + MARRIAGE, 
                  data = credit_dataset_CSV, 
                  method = "knn", trControl = train_Control,
                  tuneLength = 10)
knn_caret
plot(knn_caret)

# SVM with Linear Kernel
svm_linear_caret = train(default.payment.next.month ~ LIMIT_BAL + PAY_AMT1 + PAY_AMT3 + BILL_AMT1 + PAY_0 + PAY_2 + PAY_3 + SEX + EDUCATION + MARRIAGE, 
                         data = credit_dataset_CSV, 
                         method = "svmLinear",
                         trControl = train_Control,
                         tuneLength = 10) 
print(svm_linear_caret)

# SVM with radial 
svm_radial_caret = train(default.payment.next.month ~ LIMIT_BAL + PAY_AMT1 + PAY_AMT3 + BILL_AMT1 + PAY_0 + PAY_2 + PAY_3 + SEX + EDUCATION + MARRIAGE, 
                         data = credit_dataset_CSV, 
                         method = "svmRadial",
                         trControl = train_Control,
                         tuneLength = 10) 
print(svm_radial_caret)

# Random Forest
rf_caret = train(default.payment.next.month ~ LIMIT_BAL + PAY_AMT1 + PAY_AMT3 + BILL_AMT1 + PAY_0 + PAY_2 + PAY_3 + SEX + EDUCATION + MARRIAGE, 
                 data = credit_dataset_CSV, 
                 method = "rf",
                 trControl = train_Control,
                 tuneLength = 10) 
print(rf_caret)

# Extract accuracies from the models
accuracy_knn <- max(knn_caret$results$Accuracy)
accuracy_svm_linear <- max(svm_linear_caret$results$Accuracy)
accuracy_svm_radial <- max(svm_radial_caret$results$Accuracy)
accuracy_rf <- max(rf_caret$results$Accuracy)

   # Combine accuracies into a named vector
   model_accuracies <- c(
         kNN = accuracy_knn,
         SVM_Linear = accuracy_svm_linear,
         SVM_Radial = accuracy_svm_radial,
         Random_Forest = accuracy_rf
    )

# Barplot with accuracy labels
bar_positions <- barplot(
  model_accuracies, 
  main = "Model Comparison - Accuracies",
  xlab = "Models",
  ylab = "Accuracy",
  col = "skyblue",
  ylim = c(0, 1.1),  # Adjust ylim to leave space for labels
  las = 2
)

# Add accuracy labels above the bars
text(
  x = bar_positions, 
  y = model_accuracies + 0.2,  # Slightly above the bars
  labels = round(model_accuracies*100, 2),  # Round accuracy values to 4 decimal places
  col = "black",
  cex = 0.8  # Adjust the text size
)
