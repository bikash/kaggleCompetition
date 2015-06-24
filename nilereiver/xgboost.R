
setwd("/Users/bikash/repos/kaggle/kaggleCompetition/nilereiver/")
require(xgboost)

print("Preparing Data")
train <- read.csv('data/train.csv',header=TRUE,stringsAsFactors = F)
test <- read.csv('data/test.csv',header=TRUE,stringsAsFactors = F)
weather <- read.csv('data/weather.csv',header=TRUE,stringsAsFactors = F)


#Shuffle
train <- train[sample(nrow(train)),]

y = train$WnvPresent

train$WnvPresent     <- NULL
test$Id              <- NULL
train$NumMosquitos    <- NULL

trainlength = nrow(train)

x = rbind(train, test)

#x$Year <- as.numeric(lapply(strsplit(x$Date, "-"), function(x) x[1]))
x$month <- as.numeric(lapply(strsplit(x$Date, "-"), function(x) x[2]))
%x$week <- as.numeric(strftime(x$Date, format="%W"))%"%
x$dYear <- as.numeric(strftime(x$Date, format="%Y"))



x$Date = as.Date(x$Date, format="%Y-%m-%d")
xsDate = as.Date(paste0(x$dYear, "0101"), format="%Y%m%d")
x$dWeek = as.numeric(paste(floor((x$Date - xsDate + 1)/7)))



#x$month   		 <- x$Month
#x$week 				 <- x$Week
#x$year               <- x$Year
x$restuans           <- x$Species == 'CULEX RESTUANS'
x$pipiens            <- x$Species == 'CULEX PIPIENS'
x$both               <- x$Species == 'CULEX PIPIENS/RESTUANS'
x$territans          <- x$Species == 'CULEX TERRITANS'
x$erraticus          <- x$Species == 'CULEX ERRATICUS'
x$latitude           <- x$Latitude
x$longitude          <- x$Longitude
x$block              <- x$Block

#x$Week <- NULL
#x$Month <- NULL
#x$Year <- NULL
x$Date <- NULL
x$NumMosquitos <- NULL
x$Species <- NULL
x$Latitude <- NULL
x$Longitude <- NULL
x$Address <- NULL
x$Block <- NULL
x$AddressNumberAndStreet <- NULL
x$Trap <- NULL
x$NumMosquitos <- NULL
x$Street <- NULL

x = as.matrix(x)

print("Training the model")

param <- list("objective" = "binary:logistic",
              "eval_metric" = "auc",
              "nthread" = 16,
              "eta" = .005,
              "max_depth" = 10,
              "lambda_bias" = 0,
              "gamma" = .8,
              "min_child_weight" = 3,
              "subsample" = .9,
              "colsample_bytree" = .45,
              "scale_pos_weight" = sum(y==0) / sum(y==1))

nround = 200
# Run Cross Valication

bst.cv = xgb.cv(param=param, data = x[1:trainlength,], label = y, nfold = 10, nrounds=nround)

bst = xgboost(param=param, data = x[1:trainlength,], label = y, nrounds=nround, verbose = 2)

print("Making prediction")
pred = predict(bst, x[(nrow(train)+1):nrow(x),])
pred = matrix(pred,1,length(pred))
pred = t(pred)

print("Storing Output")
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred), pred)
names(pred) = c('Id', 'WnvPresent')
write.csv(pred, file="output/nile-river.csv", quote=FALSE,row.names=FALSE)


## Random Forest

library(party)
x1 <- x[1:trainlength,]
rf.data <-data.frame(WnvPresent=y, AddressAccuracy=x1[,1], dWeek=x1[,2], restuans=x1[,3], pipiens=x1[,4], both=x1[,5],
                     territans=x1[,6], erraticus=x1[,7], latitude=x1[,8], longitude=x1[,9], block=x1[,10])
rf = cforest(WnvPresent ~., data = rf.data, controls=cforest_unbiased(ntree=2000))

#Make a Prediction
x2 <- as.data.frame(x[-c(1:trainlength), ])
rf.pred = predict(rf, x2, OOB=TRUE, type = "response")


id<-test[,1]
submission<-cbind(id,rf.pred)
colnames(submission)[2] <- "WnvPresent"
colnames(submission)[1] <- "Id"
write.csv(submission, "output/conditional_forest_imp.csv", row.names = FALSE, quote = FALSE)

## GBM
#install.packages('gbm')he
library(gbm)
# Set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(17)
# Begin recording the time it takes to create the model
ptm5 <- proc.time()
# Create a random forest model using the target field as the response and all 93 features as inputs (.)
fit5 <- gbm(WnvPresent ~ ., data=rf.data, distribution="multinomial", n.trees=1000, 
            shrinkage=0.05, interaction.depth=12, cv.folds=2)
# Finish timing the model
fit5.time <- proc.time() - ptm5
# Test the boosting model on the holdout test dataset
trees <- gbm.perf(fit5)
fit5.stest <- predict(fit5, stest, n.trees=trees, type="response")
fit5.stest <- as.data.frame(fit5.stest)
fit5.stest.pred <- rep(NA,2000)
for (i in 1:nrow(stest)) {
  fit5.stest.pred[i] <- colnames(fit5.stest)[(which.max(fit5.stest[i,]))]}
fit5.pred <- as.factor(fit5.stest.pred)
# Create a confusion matrix of predictions vs actuals
table(fit5.pred,stest$target)
# Determine the error rate for the model
fit5$error <- 1-(sum(fit5.pred==stest$target)/length(stest$target))
fit5$error

