  #setwd("D:\\R-4.4.1")
  #install the packages if you don't have
  
  library(caret)
  library(randomForest)
  library(pROC)
  library(kernlab)
  library(corrplot)
  
  #Changes based on where you have saved the data set on your device
  
  data<-read.csv("D:\\Downloads\\diabetes_binary_health_indicators.csv")
  head(data,10)
  
  
  
  
#------------------------------------------------------------------------------------------------------------------------------------------------------------  
 
  
  
  
  #Preprocessing 
  
  #This function tells us about the internal structure of our data frame
  str(data)
  
  #We test if our data set has any missing values
  nrow(data[is.na(data),])
  
  #we see that there are over 4000 missing values out of the 5 million cells.
  
  #We replace the missing values with the column median. We use median instead of average as most of our attributes are factors
  #For the numerical attribute like BMI, they follow a normal distribution and hence the mean in close to the median
  
  replace_na<- function(df) 
  {
    df[] <- lapply(df, function(column) 
    {
      if (is.numeric(column)) 
      {
        column[is.na(column)] <- median(column, na.rm = TRUE)
      }
      column
    })
    return(df)
  }
  
  data <- replace_na(data)
  
  #We test if the funstion worked
  nrow(data[is.na(data),])
  
  
  #columns that need to be changed to factors are changed
  
  data$Sex<-as.factor(data$Sex)
  data$HighBP <-as.factor(data$HighBP )
  data$HighChol<-as.factor(data$HighChol)
  data$CholCheck<-as.factor(data$CholCheck)
  data$Smoker<-as.factor(data$Smoker)
  data$Stroke<-as.factor(data$Stroke)
  data$HeartDiseaseorAttack<-as.factor(data$HeartDiseaseorAttack)
  data$PhysActivity<-as.factor(data$PhysActivity)
  data$Fruits<-as.factor(data$Fruits)
  data$HvyAlcoholConsump<-as.factor(data$HvyAlcoholConsump)
  data$AnyHealthcare<-as.factor(data$NoDocbcCost)
  data$DiffWalk<-as.factor(data$DiffWalk)
  data$GenHlth<-as.factor(data$GenHlth)
  data$Education <-as.factor(data$Education )
  data$Income<-as.factor(data$Income)
  data$NoDocbcCost<-as.factor(data$NoDocbcCost)
  data$Age<-as.factor(data$Age)
  data$Veggies<-as.factor(data$Veggies)
  
  
  
  #outcome is changed from 0,1 to healthy and diabetic to prevent any confusion with the results. 
  
  data$Diabetes_binary[data$Diabetes_binary == "0"] <- "Healthy"
  data$Diabetes_binary[data$Diabetes_binary == "1"] <- "Diabetic"
  data$Diabetes_binary<-as.factor(data$Diabetes_binary)
  str(data)
  
 
  
  #we save the processed data set for use in the future or for other projects
  write.csv(data, "Diabetes indicators processed", row.names=FALSE)
  
  #it took about an hour for the SVM to train 10000 samples with 10 fold cross validation. Assuming complexity is O(n^2) it will take about 26 days to run this for all 250000 samples. Hence we cut short the data to 50000 samples 
  
  
  
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  
  set.seed(101)
  
  # Out of 250000+ data points only 35346 are diabetic. This gives us a ratio of 1:7 diabetic to healthy. If we train our data on a random sample of 50000 datapoints it will overfit for healthy, hence we must use undersampling ie. take 25000 diabetic and healthy samples to train our model on
 
  #Sorting based on healthy or diabetic
  data<-data[order(data[[1]]),]
  head(data)
  
  #creating a data set of only the diabetic samples
  diabetic.data.1<-data[1:35346,]
  
  #Out of the 35346 randomly chose 25000
  diabetic.data.2<-diabetic.data.1[sample(nrow(diabetic.data.1),25000),]
  
  #The remaining ones are added into a test data set. We want 10000 data points in our test data and we want to preserve the ratio 1:7 ie 1250 should be diabetic and 8750 healthy.
  test.data.1 <- diabetic.data.1[!rownames(diabetic.data.1) %in% rownames(diabetic.data.2), ]
  test.data.1<-test.data.1[sample(nrow(test.data.1),1250),]
  
  #we repeat the same for healthy class
  healthy.data.1 <- data[!rownames(data) %in% rownames(diabetic.data.1), ]
  healthy.data.2<-healthy.data.1[sample(nrow(healthy.data.1),25000),]
  
  test.data.2 <- healthy.data.1[!rownames(healthy.data.1) %in% rownames(healthy.data.2), ]
  test.data.2<-test.data.2[sample(nrow(test.data.2),8750),]
  
  # we combine the test data sets with diabetic and healthy classes into one
  test.data<-rbind(test.data.1,test.data.2)
  
  test.data <- sample(test.data)
  
  #train data with 25000 diabetic and 25000 healthy samples
  train.data<-rbind(diabetic.data.2,healthy.data.2)
  train.data<-sample(train.data)
  
  
  
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  
  
# This is our first test to check which attributes play a more important role. 

# we perform cross tabulation of diabetes binary against all the other attributes.If the ratio of diabetic to healthy changes as the attribute changes indicates effectivness of that attribute.

for (col_name in names(data)[-1]) 
{
  print(col_name)
  print(xtabs(~ Diabetes_binary + data[[col_name]], data = data))
}

# we see many attributes like sex, healthcare, fruits, veggies etc has similar rations. We run a logistic regression using only one attribute at a time and compare AIC scores

aic_scores <- c()
col_names <- c()  

for (col_name in names(train.data)[-1]) 
{
  print(col_name)
  
  logistic <- glm(Diabetes_binary ~ train.data[[col_name]], data = train.data, family = "binomial")
   
  aic_scores <- c(aic_scores, AIC(logistic))
  col_names <- c(col_names, col_name)
  
  print(logistic)
}


aic_data <- data.frame(Column = col_names, AIC = aic_scores)

#plotting the AIC scores of the different attributes. 

ggplot(aic_data, aes(x = reorder(Column, AIC), y = AIC)) + geom_bar(stat = "identity", fill = "skyblue") + coord_flip() + labs(title = "AIC Scores for Logistic Regressions by Attribute", x = "Column", y = "AIC Score") + theme_minimal()

# we see that NoDocbcCost, Sec, AnyHealthcare, Fruits and Veggies have very high scores

#Logistic regression using all attributes

logistic<-glm(Diabetes_binary~., data=train.data, family="binomial")
summary(logistic)

#we see many attributes without a star marked against them which suggests they might not good for training.

#we remove attributes that didn't perform well in the 3 different tests.

train.data.2 <- train.data[, !(names(train.data) %in% c("Education", "Veggies", "Fruits", "AnyHealthcare", "MentHlth","Smoker","NoDocbcCost"))]


# we train the model with train.data and train.data.2 and compare results

#train.data
logit_model <- train(Diabetes_binary ~ .,data = train.data, method = "glm",family = binomial)

summary(logit_model)

logistic_preds <- predict(logit_model, newdata = test.data)


logit_cm<-confusionMatrix(logistic_preds, test.data$Diabetes_binary)
print(logit_cm)

confusion_matrix <- function(cm, model_name) 
{
  cm_data <- as.data.frame(cm$table)
  cm_data$Model <- model_name
  cm_data$Prediction <- factor(cm_data$Prediction, levels = rev(levels(cm_data$Prediction)))
  ggplot(cm_data, aes(x = Reference, y = Prediction)) + geom_tile(aes(fill = Freq), color = "black") + scale_fill_gradient(low = "red", high = "green") + geom_text(aes(label = Freq), vjust = 1) + labs(title = paste("Confusion Matrix for", model_name), x = "Actual", y = "Predicted") + theme_minimal() + theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot confusion matrix
confusion_matrix(logit_cm, "Logistic Regression")


#train.data.2
logit_model <- train(Diabetes_binary ~ .,data = train.data.2, method = "glm",family = binomial)

summary(logit_model)

logistic_preds <- predict(logit_model, newdata = test.data)

logit_cm<-confusionMatrix(logistic_preds, test.data$Diabetes_binary)
print(logit_cm)


# Plot confusion matrix
confusion_matrix(logit_cm, "Logistic Regression")

# we see a small increase in sensitivity, accuracy and precision. Hence we use the modified data set. 

train.data<-train.data.2



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




#Define the 10-fold cross-validation method
cv <- trainControl(method = "cv",number = 10,savePredictions = "final",classProbs = TRUE,summaryFunction = twoClassSummary)  


#Train Logistic Regression model
logit_model <- train(Diabetes_binary ~ .,data = train.data, method = "glm",family = binomial,trControl = cv,metric = "ROC")
summary(logit_model)


#Train Decision Tree model
dt_model <- train(Diabetes_binary ~ ., data = train.data, method = "rpart", trControl = cv)
summary(dt_model)

#Train Random Forest model
rf_model <- train(Diabetes_binary ~ .,data = train.data, method = "rf",trControl = cv,metric = "ROC",tuneGrid = expand.grid(mtry = sqrt(ncol(data) - 1)),  ntree = 500)
summary(rf_model)


#Train Support Vector Machine model
svm_model <- train(Diabetes_binary ~ .,data = train.data, method = "svmRadial",trControl = cv,metric = "ROC",tuneLength = 5)
print(svm_model)




#Predict classes on test.data
logistic_preds <- predict(logit_model, newdata = test.data)
dt_preds <- predict(dt_model, newdata = test.data)
rf_preds <- predict(rf_model, newdata = test.data)
svm_preds <- predict(svm_model, newdata = test.data)



#Predict probabilities on test.data
logistic_probs <- predict(logit_model, newdata = test.data, type = "prob")[,2]
dt_probs <- predict(dt_model, newdata = test.data, type = "prob")[,2]
rf_probs <- predict(rf_model, newdata = test.data, type = "prob")[,2]
svm_probs <- predict(svm_model, newdata = test.data, type = "prob")[,2]




# Confusion matrices for the 4 models

logit_cm<-confusionMatrix(logistic_preds, test.data$Diabetes_binary)
print(logit_cm)


dt_cm<-confusionMatrix(dt_preds, test.data$Diabetes_binary)
print(dt_cm)


rf_cm<-confusionMatrix(rf_preds, test.data$Diabetes_binary)
print(rf_cm)


svm_cm<-confusionMatrix(svm_preds, test.data$Diabetes_binary)
print(svm_cm)



# Plot confusion matrices using the function we defined before. 

confusion_matrix(logit_cm, "Logistic Regression")
confusion_matrix(dt_cm, "Decision Tree")
confusion_matrix(rf_cm, "Random Forest")
confusion_matrix(svm_cm, "SVM")


#we compare the sensitivity, precision and accuracy of the 4 models

sensitivty_scores<-c()
precision_scores<-c()
specificity_scores<-c()
accuracy_scores<-c()
models<-c("logistic regression","Decision Tree", "Random FOrest", "SVM")


#Retrieving the data from the confusion matrix



logit_sensitivty <- logit_cm$byClass["Sensitivity"]
logit_precision <- logit_cm$byClass["Pos Pred Value"]
logit_specificity <- logit_cm$byClass["Specificity"]
logit_accuracy <- logit_cm$overall["Accuracy"]
sensitivty_scores<-c(sensitivty_scores, logit_sensitivty)
precision_scores<-c(precision_scores, logit_precision)
specificity_scores<-c(specificity_scores, logit_specificity)
accuracy_scores<-c(accuracy_scores, logit_accuracy)

dt_sensitivty <- dt_cm$byClass["Sensitivity"]
dt_precision <- dt_cm$byClass["Pos Pred Value"]
dt_specificity <- dt_cm$byClass["Specificity"]
dt_accuracy <- dt_cm$overall["Accuracy"]
sensitivty_scores<-c(sensitivty_scores, dt_sensitivty)
precision_scores<-c(precision_scores, dt_precision)
specificity_scores<-c(specificity_scores, dt_specificity)
accuracy_scores<-c(accuracy_scores, dt_accuracy)

rf_sensitivty <- rf_cm$byClass["Sensitivity"]
rf_precision <- rf_cm$byClass["Pos Pred Value"]
rf_specificity <- model_cm$byClass["Specificity"]
rf_accuracy <- rf_cm$overall["Accuracy"]
sensitivty_scores<-c(sensitivty_scores, rf_sensitivty)
precision_scores<-c(precision_scores, rf_precision)
specificity_scores<-c(specificity_scores, rf_specificity)
accuracy_scores<-c(accuracy_scores, rf_accuracy)

svm_sensitivty <- svm_cm$byClass["Sensitivity"]
svm_precision <- svm_cm$byClass["Pos Pred Value"]
svm_specificity <- svm_cm$byClass["Specificity"]
svm_accuracy <- svm_cm$overall["Accuracy"]
sensitivty_scores<-c(sensitivty_scores, svm_sensitivty)
precision_scores<-c(precision_scores, svm_precision)
specificity_scores<-c(specificity_scores, svm_specificity)
accuracy_scores<-c(accuracy_scores, svm_accuracy)



#Plotting sensitivity values of the 4 models
sensitivity.df= data.frame(Model= models, sensitivty_scores)
dotplot(Model ~ sensitivty_scores, data = sensitivity.df, main = "Sensitivity Comparison", xlab = "Sensitivity", ylab = "Model", pch = 10,  cex=2, col = "black")

#Plotting precision values of the 4 models
precision.df= data.frame(Model=models, precision_scores)
dotplot(Model ~ precision_scores, data = precision.df, main = "Precision Comparison", xlab = "Precision", ylab = "Model", pch = 10,  cex=2, col = "black")

#Plotting specificity values of the 4 models
specificity.df= data.frame(Model=models, specificity_scores)
dotplot(Model ~ specificity_scores, data = specificity.df, main = "Specificity Comparison", xlab = "Specificity", ylab = "Model", pch = 10,  cex=2, col = "black")


#Plotting accuracy values of the 4 models
accuracy.df= data.frame(Model=models, accuracy_scores)
dotplot(Model ~ accuracy_scores, data = accuracy.df, main = "Accuracy Comparison", xlab = "Accuracy", ylab = "Model", pch = 10,  cex=2, col = "black")




#Plotting ROC curves from probability values using pROC

roc_logistic <- roc(test.data$Diabetes_binary, logistic_probs)
roc_dt <- roc(test.data$Diabetes_binary, dt_probs)
roc_rf <- roc(test.data$Diabetes_binary, rf_probs)
roc_svm <- roc(test.data$Diabetes_binary, svm_probs)

plot(roc_logistic, col = "blue", main = "ROC Curves", legacy.axes = TRUE)
plot(roc_dt, col = "yellow", add = TRUE)
plot(roc_rf, col = "green", add = TRUE)
plot(roc_svm, col = "red", add = TRUE)
legend("bottomright", legend = models,
       col = c("blue", "yellow", "green", "red"), lwd = 6)
  

#Retrieving the AUC values of the 4 models

logistic_auc <- auc(roc_logistic)
dt_auc <- auc(roc_dt)
rf_auc <- auc(roc_rf)
svm_auc <- auc(roc_svm)

print(paste("Logistic Regression AUC:", auc(roc_logistic)))
print(paste("Decision Tree AUC:", auc(roc_dt)))
print(paste("Random Forest AUC:", auc(roc_rf)))
print(paste("SVM AUC:", auc(roc_svm)))



auc_results <- data.frame(Model = rep((models), each = 1), AUC = c(logistic_auc,dt_auc, rf_auc, svm_auc))

#plotting the AUC values
dotplot(Model ~ AUC, data = auc_results,main = "AUC Comparison", xlab = "AUC", ylab = "Model", pch = 10,  cex=2, col = "blue")


