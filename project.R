library(glmnet)
library(ROCR)
library(randomForest)
df = read.csv('data/santander.csv')

# Remove ID_Code Column
df = df[-1]

# Sample 1000 rows from df
set.seed(75)
data = df[sample(nrow(df), 4000),]
# Reset indexes of df
row.names(data) <- NULL

n = dim(data)[1]
p = dim(data)[2] - 1

X = as.matrix(data[,-1])
y = data$target

## For final run - set iterations to 50
iterations                   = 50

# Resulting data frames
lasso.results                = data.frame(method="LASSO", matrix(ncol=3, nrow=iterations))
elast.results                = data.frame(method="ELAST", matrix(ncol=3, nrow=iterations))
ridge.results                = data.frame(method="RIDGE", matrix(ncol=3, nrow=iterations))
randf.results                = data.frame(method="RANDF", matrix(ncol=3, nrow=iterations))
column.Names                 = c("METHOD","AUC.TRAIN", "AUC.TEST", "FITTING.TIME")
colnames(lasso.results)      = column.Names
colnames(elast.results)      = column.Names
colnames(ridge.results)      = column.Names
colnames(randf.results)      = column.Names

# For coeffient plots
lasso.coeff.mat = data.frame(matrix(ncol=length(colnames(X.train)), nrow=1))
elasticnet.coeff.mat = data.frame(matrix(ncol=length(colnames(X.train)), nrow=1))
ridge.coeff.mat = data.frame(matrix(ncol=length(colnames(X.train)), nrow=1))
rf.coeff.mat = data.frame(matrix(ncol=length(colnames(X.train)), nrow=1))
colnames(lasso.coeff.mat) = c(colnames(X.train))
colnames(elasticnet.coeff.mat) = c(colnames(X.train))
colnames(ridge.coeff.mat) = c(colnames(X.train))
colnames(rf.coeff.mat) = c(colnames(X.train))

# Settings for cv.fit plots
par(mfrow=c(3,1))

## Run for Lasso, Elastic, Ridge
alphas = c(1, 0.5, 0)

for (run in seq(iterations)) {
  runStart      = Sys.time()
  # Take 90/10 split
  sample.idx    = sample(nrow(data), size = n * 0.9)
  X.train       = X[sample.idx,]
  X.test        = X[-sample.idx,]
  y.train       = y[sample.idx]
  y.test        = y[-sample.idx]
  
  weight_n = dim(X.train)[1]
  p = dim(X.train)[2]
  n.P                 =        sum(y.train)
  n.N                 =        weight_n - n.P
  ww                  =        rep(1,weight_n)
  ww[y.train==1]      =        n.N/n.P
  
  # Logistic Regression
  for (a in alphas){
    logisticType = ""
    if (a==1)  { logisticType = "LASSO"}
    if (a==0.5){ logisticType = "ELAST"}
    if (a==0)  { logisticType = "RIDGE"}
  
    # Fitting model - To do - Try adding in weights
    start         = Sys.time()
    cv.fit        = cv.glmnet(X.train, y.train, family="binomial", alpha=a, type.measure="class", nfolds=10, weights = ww, intercept=TRUE)
    fit           = glmnet(X.train, y.train, family="binomial", alpha=a, lambda=cv.fit$lambda.min, wegihts=ww, intercept=TRUE)
    end           = Sys.time()
    time          = end - start
    
    beta0.hat     = fit$a0
    beta.hat      = fit$beta
    prob.train    = predict(fit, newx = X.train, type="response")
    prob.test     = predict(fit, newx = X.test, type="response")
    
    # Calculate train AUC and test AUC
    pred.train           <- prediction(c(prob.train), y.train)
    auc.ROCR.train       <- performance(pred.train, measure="auc")
    auc.train            <- auc.ROCR.train@y.values[[1]]
    pred.test            <- prediction(c(prob.test), y.test)
    auc.ROCR.test        <- performance(pred.test, measure="auc")
    auc.test             <- auc.ROCR.test@y.values[[1]]
    
    # Populate results data frame
    if (logisticType == "LASSO")      {lasso.results[run,2:4]       <- c(auc.train, auc.test, time)}
    if (logisticType == "ELAST")      {elast.results[run,2:4]       <- c(auc.train, auc.test, time)}
    if (logisticType == "RIDGE")      {ridge.results[run,2:4]       <- c(auc.train, auc.test, time)}

    if (run == iterations && logisticType == "LASSO")      {lasso.coeff.mat[1,]       <- t(beta.hat)}
    if (run == iterations && logisticType == "ELAST")      {elasticnet.coeff.mat[1,]  <- t(beta.hat)}
    if (run == iterations && logisticType == "RIDGE")      {ridge.coeff.mat[1,]       <- t(beta.hat)}

    # Print cv.fit plot for logistic method on last iteration
    if (run == iterations) {plot(cv.fit)}
    
    # Screen progress output
    print(sprintf("Run %i: %s Fitting Runtime: %3.4f seconds, Train AUC: %.4f Test AUC: %.4f", run, logisticType, time, auc.train, auc.test))
  }
  
  # Random forest
  dat.train     = data.frame(X.train, y.train=as.factor(y.train))
  dat.test      = data.frame(X.test, y.test=as.factor(y.test))
  start         = Sys.time()
  rf.fit        = randomForest(y.train~., data = dat.train, mtry = sqrt(p), do.trace=TRUE, ntree=50, classwt=rev(unique(ww)), importance=TRUE)
  end           = Sys.time()
  time          = end - start
  
  prob.train = predict(rf.fit, dat.train, type="prob")
  prob.test  = predict(rf.fit, dat.test, type="prob")
  
  # Calculate train AUC and test AUC
  pred.train             <- prediction(as.vector(prob.train[,2]), y.train)
  auc.ROCR.train         <- performance(pred.train, measure="auc")
  auc.train              <- auc.ROCR.train@y.values[[1]]
  
  pred.test              <- prediction(as.vector(prob.test[,2]), y.test)
  auc.ROCR.test          <- performance(pred.test, measure="auc")
  auc.test               <- auc.ROCR.test@y.values[[1]]
  
  # Populate results
  randf.results[run,2:4] <- c(auc.train, auc.test, time)
  print(sprintf("Run %i: %s Fitting Runtime: %3.4f seconds, Train AUC: %.4f Test AUC: %.4f", run, "RANDF", time, auc.train, auc.test))
  
  # Iteration completed, printing total time for run
  runEnd        = Sys.time()
  runTime       = runEnd - runStart
  print(sprintf("Run %i took %3.4f seconds", run, runTime))
}

## Final dataframe
finalResults = rbind(lasso.results,elast.results,ridge.results, randf.results)

## AUC Train/Test Boxplot per algorithm
bp           <- finalResults %>% 
                  gather(AUC.TYPE, AUC, c(AUC.TRAIN,AUC.TEST))
bp$METHOD    <- factor(bp$METHOD, levels=c("LASSO","ELAST","RIDGE","RANDF"))
bp$AUC.TYPE  <- factor(bp$AUC.TYPE, levels=c("AUC.TRAIN","AUC.TEST"))

ggplot(bp, aes(x=AUC.TYPE, y=AUC, fill=METHOD)) +
  geom_boxplot() +
  guides() +
  theme_bw()

# Showing importance of RF predictors based on GINI and Accuracy
rfImportance <- importance(rf.fit)
varImpPlot(rf.fit)

## Calculating TPR and FPR for one run and generating ROC plot
vec.theta             <- seq(0,1,by=0.01)
vec.theta.len         <- length(vec.theta)
mat.tpr.fpr.train     <- matrix(0, nrow=vec.theta.len, ncol=3)
mat.tpr.fpr.train[,1] <- vec.theta
mat.tpr.fpr.test      <- matrix(0, nrow=vec.theta.len, ncol=3)
mat.tpr.fpr.test[,1]  <- vec.theta
for (i in 1:vec.theta.len){
  # Train dataset
  y.hat.train                  = ifelse(prob.train > vec.theta[i], 1, 0)
  FP.train                     = sum(y.train[y.hat.train==1] == 0)
  TP.train                     = sum(y.train[y.hat.train==1] == 1)
  P.train                      = sum(y.train == 1)
  N.train                      = sum(y.train == 0)
  FPR.train                    = FP.train / N.train
  TPR.train                    = TP.train / P.train
  mat.tpr.fpr.train[i,c(2,3)]  = c(TPR.train, FPR.train)
  
  # Test dataset
  y.hat.test                   = ifelse(prob.test > vec.theta[i], 1, 0)
  FP.test                      = sum(y.test[y.hat.test==1] == 0)
  TP.test                      = sum(y.test[y.hat.test==1] == 1)
  P.test                       = sum(y.test == 1)
  N.test                       = sum(y.test == 0)
  FPR.test                     = FP.test / N.test
  TPR.test                     = TP.test / P.test
  mat.tpr.fpr.test[i,c(2,3)]   = c(TPR.test, FPR.test)
}

## Coeffient plot
lasso.coeff.mat = lasso.coeff.mat[, order(elasticnet.coeff.mat, decreasing = TRUE)]
elasticnet.coeff.mat = elasticnet.coeff.mat[, order(elasticnet.coeff.mat, decreasing = TRUE)]
ridge.coeff.mat = ridge.coeff.mat[, order(elasticnet.coeff.mat, decreasing = TRUE)]
rf.coeff.mat = rf.coeff.mat[, order(elasticnet.coeff.mat, decreasing = TRUE)]
all.coeff.mat = rbind(lasso.coeff.mat, elasticnet.coeff.mat, ridge.coeff.mat, rf.coeff.mat)
all.coeff.mat = cbind(seq(1:200), t(all.coeff.mat))
colnames(all.coeff.mat) = c('Elastic Net Ordered Index', 'Lasso', 'Elastic_Net', 'Ridge', 'Random_Forest')
all.coeff.mat = as.data.frame(all.coeff.mat)

par(mfrow=c(4,1))
par(mar = c(2, 4, 2, 2))
barplot(all.coeff.mat$Elastic_Net, horiz=FALSE, main = 'Elastic Net', ylab = 'Coefficient Value', col = 'blue')
barplot(all.coeff.mat$Lasso, horiz=FALSE, main = 'Lasso', ylab = 'Coefficient Value', col = 'gray')
barplot(all.coeff.mat$Ridge, horiz=FALSE, main = 'Ridge', ylab = 'Coefficient Value', col = 'red')
barplot(all.coeff.mat$Random_Forest, horiz=FALSE, main = 'Random Forest', ylab = 'MeanDecreaseGini', col = 'white')
