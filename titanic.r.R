#############################
#                           #
#       RMS   TITANIC       #
# Prediction of Survival of #
#   Passengers on Titanic   #
#############################

#   DECISION TREE, CONFUSION MATRIX AND SVM#  


# Machine Learning
# Shivam Panchal


###################################################################################
#                                                                                 #
#  For the following Project, we would like to perform a sample data exploration  #
#  based on the dataset of passengers surviving the Titanic shipwreck. The steps  #
#  we demonstrate here follow how to collect data from the online source, Kaggle; #
#  clean data through data munging; perform basic exploratory data analysis to    #
#  discover important attributes that might give a prediction of the survival     #
#  rate; perform advanced exploratory data analysis using the classification      #
#  algorithm to predict the survival rate of the given data; and finally,         #
#  perform model assessment to generate a prediction model.                       #
#                                                                                 #
###################################################################################

## Step 1: Loading the data and manipulation of data

train <- read.csv("train.csv", na.strings = c("NA",""))
test <- read.csv("test.csv", na.strings = c("NA",""))

str(train)
str(test)

# Converting the Survived column into a factor
train$Survived <- factor(train$Survived)

# finding the missing values
is.na(train$Age)
sum(is.na(train$Age))
sum(is.na(train$Age))/length(train$Age)

#  viewing the percentage of missing data, in each column
sapply(train,function(df){
  sum(is.na(df)==T)/length(df);
 })

# one may also use the Amelia package to visualize the missing values
# install.packages("Amelia")

library(Amelia)
missmap(train, main = "Missing Map")

# Okay, now we have learnt about the condition of data and have understood it.
# We will now, impute the missing data within each required attribute since they will have a significant effect on the conclusion.

# First, list the distribution of Port of Embarkation. Here, we add the useNA = "always" argument to show the number of NA values contained within train dataset

table(train$Embarked, useNA = "always")
#  C    Q    S   <NA> 
# 168   77  644    2 

# Assign the two missing values to a more probable port (that is, the most counted port), which is Southampton in this case: 
train$Embarked[which(is.na(train$Embarked))] = 'S'
table(train$Embarked, useNA = "always")
#  C    Q    S    <NA> 
#  168   77  646    0 


# In order to discover the types of titles contained in the names of train.data, we first tokenize 
# train.data$Name by blank (a regular expression pattern as "\\s+"), and then count the frequency 
# of occurrence with the table function. After this, since the name title often ends with a period, 
# we use the regular expression to grep the word containing the period. In the end, sort the table 
# in decreasing order: 

train$Name <- as.character(train$Name)
table_prefixes <- table(unlist(strsplit(train$Name, "\\s+")))
sort(table_prefixes [grep('\\.',names(table_prefixes))], decreasing=TRUE)

#  To obtain which title contains missing values, you can use str_match provided by the stringr package
#  to get a substring containing a period, then bind the column together with cbind. Finally, by using 
#  the table function to acquire the statistics of missing values, you can work on counting each title: 

library(stringr)
tb = cbind(train$Age, str_match(train$Name, " [a-zA-Z]+\\."))
table(tb[is.na(tb[,1]),2])

# For a title containing a missing value, one way to impute data is to assign the mean value for each 
# title (not containing a missing value): 

# grepl: pattern matching and replacement
mean.mr = mean(train$Age[grepl(" Mr\\.", train$Name) & !is.na(train$Age)])
mean.mrs = mean(train$Age[grepl(" Mrs\\.", train$Name) & !is.na(train$Age)])
mean.dr = mean(train$Age[grepl(" Dr\\.", train$Name) & !is.na(train$Age)])
mean.miss = mean(train$Age[grepl(" Miss\\.", train$Name) & !is.na(train$Age)])
mean.master =  mean(train$Age[grepl(" Master\\.", train$Name) & !is.na(train$Age)])

# Then, assign the missing value with the mean value of each title
train$Age[grepl(" Mr\\.", train$Name) & is.na(train$Age)] = mean.mr
train$Age[grepl(" Mrs\\.", train$Name) & is.na(train$Age)] = mean.mrs
train$Age[grepl(" Dr\\.", train$Name) & is.na(train$Age)] = mean.dr
train$Age[grepl(" Miss\\.", train$Name) & is.na(train$Age)] = mean.miss
train$Age[grepl(" Master\\.", train$Name) & is.na(train$Age)] = mean.master


## STEP 2

# Exploration and vizualization of data
barplot(table(train$Survived), main = "Passenger Survival", names = c("Perished","Survived"))
barplot(table(train$Pclass), main="Passenger Class",  names= c("first", "second", "third"))
barplot(table(train$Sex), main = "Passenger Gender")
barplot(table(train$SibSp), main="Passenger Siblings")
barplot(table(train$Parch), main="Passenger Parch")
barplot(table(train$Embarked), main="Port of Embarkation")

hist(train$Fare, main="Passenger Fare", xlab = "Fare")
hist(train$Age, main = "Passenger Age" , xlab = "Age")


counts = table( train$Survived, train$Sex)
barplot(counts,  col=c("darkblue","red"), legend = c("Perished", "Survived"), main = "Passenger Survival by Sex")

counts = table( train$Survived, train$Pclass) 
barplot(counts,  col=c("darkblue","red"), legend =c("Perished", "Survived"), main= "Passenger Survival by passenger Class" )

counts = table(train$Sex, train$Pclass)
barplot(counts, col=c("darkblue","red"), legend = rownames(counts), main= "Passenger Gender by Class")

# Histogram of Passenger Ages
hist(train$Age[which(train$Survived == "0")], main= "Survival by Passenger Age", xlab="Age", ylab="Count", col ="blue", breaks=seq(0,80,by=2))
hist(train$Age[which(train$Survived == "1")], col ="red", add = T, breaks=seq(0,80,by=2))


boxplot(train$Age ~ train$Survived,main="Passenger Survival by Age",xlab="Survived", ylab="Age")
# So, these plots help us to vizualise the data and tells, what factora are more important



# To categorize people with different ages into different groups, such as children (below 13), youths (13 to 19), adults (20 to 65), and 
# senior citizens (above 65), execute the following commands:

train.child = train$Survived[train$Age < 13]
length(train.child)
train.youth = train$Survived[train$Age >= 15 & train$Age < 25]
length(train.youth)
train.adult  = train$Survived[train$Age >= 20 & train$Age < 65]
train.adult
train.senior  = train$Survived[train$Age >= 65]
train.senior

## set the seed to make your partition reproductible
set.seed(123)
n = nrow(train)
train.Set = sample(1:n, size = round(0.7*n), replace=T)
titanic.train = train[train.Set,]
titanic.test = train[-train.Set,]



## Predicting passenger survival with a decision tree

library(party)
train.Set_ctree <- ctree(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked, data = titanic.train)
train.Set_ctree

plot(train.Set_ctree, mains = "Conditional Inferences tree of Titanic DataSet")


## Validating the power of prediction with a confusion matrix
# The assessment can be done by using the confusion matrix provided by the caret package to generate a confusion matrix, which is one method to measure the accuracy of predictions. 
library(e1071)
titanic.svm.model <- svm(Survived ~ Pclass + Sex + Age + SibSp + Fare + Parch + Embarked, data = titanic.train, probability = TRUE)
ctree.predict <- predict(train.Set_ctree, titanic.test)


library(caret)

confusionMatrix(ctree.predict, titanic.test$Survived)






