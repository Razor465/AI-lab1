# підключаємо бібліотеки

library(e1071)
library(fastAdaboost)
library(class)
library(randomForest)
library(C50)

# зчитуємо дані

data <- read.table('C:\\Users\\Razor\\Desktop\\дистанційне навчання\\статистичні алгоритми навчання\\lab1\\glass_data.DATA', 
                   header = F, sep = ',') 
data <- data[,-1] # видаляємо стовпчик з ID

# назвемо стовбчики
colnames(data) <- c('RI', 'Na', 'Mg', "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")
summary(data)

s = round(length(data[,1])*0.8)
# розіб'ємо вибірку на тренувальну і тестову
Y <- data[,10] # відгук
X <- data[,1:9] 

S <- sample(1:length(data[,1]), s)
# тренувальна вибірка
Ytrain <- Y[S]
Xtrain <- X[S,]
# тестова вибірка
Ytest <- Y[-S]
Xtest <- X[-S,]

#
# метод k-nn
#

result_knn <- knn(Xtrain, Xtest, cl = Ytrain)

table(result_knn, Ytest)
sum(diag(table(result_knn, Ytest)))/length(Ytest)

for(i in 2:20){
  r <- knn(Xtrain, Xtest, cl = Ytrain, k = i)
  print(paste('k=',i,',точність: ', sum(diag(table(r, Ytest)))/length(Ytest)))
}

#
# метод svm
#

train_sample <- data.frame(as.factor(Ytrain), Xtrain) # тренувальна вибірка
test_sample <- data.frame(as.factor(Ytest), Xtest) # тестова вибірка
colnames(train_sample)[1] <- 'Y'
colnames(test_sample)[1] <- 'Y'

model_svm <- svm(Y ~., data = train_sample, cost = 1, type = 'C-classification', kernel = 'polynomial')

result_svm <- predict(model_svm, test_sample)

table(result_svm, Ytest)
sum(diag(table(result_svm, Ytest)))/length(Ytest)

tuned_svm_model <- tune(svm, Y ~., data = train_sample, 
                        ranges = list(cost = c(0.01:50, 0.01)), 
                        kernel = c('linear', 'radial'))

tuned_svm_model
result_tuned_svm <- predict(tuned_svm_model$best.model, test_sample)
  
table(result_tuned_svm, Ytest)
sum(diag(table(result_tuned_svm, Ytest)))/length(Ytest)
  
#
# моделі, засновані на деревах
#

# randomForest
model_rF <- randomForest(Y~. , data = train_sample)
result_rF <- predict(model_rF, test_sample)

table(result_rF, Ytest)
sum(diag(table(result_rF, Ytest)))/length(Ytest)

# c5.0
model_c50 <- C5.0(Y ~. , data = train_sample)
result_c50 <- predict(model_c50, test_sample)

table(result_c50, Ytest)
sum(diag(table(result_c50, Ytest)))/length(Ytest)

#
# введемо факторну змінну
#

m <- mean(data[,9])
data_mod <- data
data_mod[data_mod[,9] >= m, 9] <- 1
data_mod[data_mod[,9] < m, 9] <- 0

# розіб'ємо на тренувальну та тестові вибірки

Y_mod <- data_mod[,10] # відгук
X_mod <- data_mod[,1:9] 

# тренувальна вибірка
Ytrain_mod <- Y_mod[S]
Xtrain_mod <- X_mod[S,]
# тестова вибірка
Ytest_mod <- Y_mod[-S]
Xtest_mod <- X_mod[-S,]

train_sample_mod <- data.frame(as.factor(Ytrain_mod), Xtrain_mod) # тренувальна вибірка
test_sample_mod <- data.frame(as.factor(Ytest_mod), Xtest_mod) # тестова вибірка
colnames(train_sample_mod)[1] <- 'Y'
colnames(test_sample_mod)[1] <- 'Y'

#
# метод k-nn
#

result_knn_mod <- knn(Xtrain_mod, Xtest_mod, cl = Ytrain_mod)

table(result_knn_mod, Ytest_mod)
sum(diag(table(result_knn_mod, Ytest_mod)))/length(Ytest_mod)

for(i in 2:20){
  r <- knn(Xtrain_mod, Xtest_mod, cl = Ytrain_mod, k = i)
  print(paste('k=',i,',точність: ', sum(diag(table(r, Ytest_mod)))/length(Ytest_mod)))
}

#
# метод svm
#

model_svm_mod <- svm(Y ~., data = train_sample_mod, cost = 1, type = 'C-classification', kernel = 'polynomial')

result_svm_mod <- predict(model_svm_mod, test_sample_mod)

table(result_svm_mod, Ytest_mod)
sum(diag(table(result_svm_mod, Ytest_mod)))/length(Ytest_mod)

tuned_svm_model_mod <- tune(svm, Y ~., data = train_sample_mod, 
                        ranges = list(cost = c(0.01:50, 0.01)), 
                        kernel = c('linear', 'radial'))

tuned_svm_model_mod
result_tuned_svm_mod <- predict(tuned_svm_model_mod$best.model, test_sample_mod)

table(result_tuned_svm_mod, Ytest_mod)
sum(diag(table(result_tuned_svm_mod, Ytest_mod)))/length(Ytest_mod)

#
# моделі, засновані на деревах
#

# randomForest
model_rF_mod <- randomForest(Y~. , data = train_sample_mod)
result_rF_mod <- predict(model_rF_mod, test_sample_mod)

table(result_rF_mod, Ytest_mod)
sum(diag(table(result_rF_mod, Ytest_mod)))/length(Ytest_mod)

# c5.0
model_c50_mod <- C5.0(Y ~. , data = train_sample_mod)
result_c50_mod <- predict(model_c50_mod, test_sample_mod)

table(result_c50_mod, Ytest_mod)
sum(diag(table(result_c50_mod, Ytest_mod)))/length(Ytest_mod)


  