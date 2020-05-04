library(randomForest)
library(glmnet)
library(car)
library(rpart)
set.seed(2222)
features <-read.csv("~/Desktop/data/molorg_features.csv")
pol <- read.csv("~/Desktop/data/molorg_pol.csv", sep="")
features $pol<-pol$pol_Bohr3 ##combining features and response
features<-na.omit(features) ##omitting all the missing values
features$Psi_i_1s<-NULL ##constant zeros

for (i in 1:10){
  plot(x=features$pol,y=features[,i])
}

m<-mean(features$pol)
std<-sqrt(var(features$pol))
hist(features$pol,xlab='response',main='distribution of response',freq=FALSE) ##following normal distribution
curve(dnorm(x, mean=m, sd=std), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

boxplot(features$pol,xlab='response',main='boxplot of response')##no outliers

set.seed(12345)
train=sample(1:nrow(features),round(0.80*nrow(features)))
test=-train
testdata=features [test,]
traindata=features [train,]

linearmodel<-lm(pol~.,traindata)
pred<-predict(linearmodel, testdata)
lm_test_RMSE<-sqrt(mean((testdata$pol-pred)^2))##19.79608
cor.test(~SpPos_B.p.+pol,
          data=features,
          method = "pearson",
          conf.level = 0.95) ##0.96304

cor.test(~Wi_B.p.+pol,
         data=features,
         method = "pearson",
         conf.level = 0.95) ##0.9628935 

cor.test(~VE2sign_L+pol,
         data=features,
         method = "pearson",
         conf.level = 0.95) ##-0.9082469 

linearmodel1<-lm(pol~SpPos_B.p.,traindata)
pred1<-predict(linearmodel1, testdata)
lm_test_RMSE1<-sqrt(mean((testdata$pol-pred1)^2))##12.14067
plot(traindata$SpPos_B.p.,traindata$pol,xlab='SpPos_B.p.',ylab='polarizability',main='fitted linear model')
abline(linearmodel1)

linearmodel2<-lm(pol~Wi_B.p.,traindata)
pred2<-predict(linearmodel2, testdata)
lm_test_RMSE2<-sqrt(mean((testdata$pol-pred2)^2))##12.14102
plot(traindata$Wi_B.p.,traindata$pol,xlab='Wi_B.p.',ylab='polarizability',main='fitted linear model')
abline(linearmodel2)

linearmodel3<-lm(pol~VE2sign_L,traindata)
pred3<-predict(linearmodel3, testdata)
lm_test_RMSE3<-sqrt(mean((testdata$pol-pred3)^2))##19.46847
plot(traindata$VE2sign_L,traindata$pol,xlab='VE2sign_L',ylab='polarizability',main='fitted linear model')
abline(linearmodel3)

linearmodel4<-lm(pol~SpPos_B.p.+Wi_B.p.+VE2sign_L,traindata)
pred4<-predict(linearmodel4, testdata)
lm_test_RMSE4<-sqrt(mean((testdata$pol-pred4)^2))##11.45094

linearmodel5<-lm(pol~SpAbs_Dz.e. +VE1sign_X+VE2sign_X +Ho_A ,traindata)
pred5<-predict(linearmodel5, testdata)
lm_test_RMSE5<-sqrt(mean((testdata$pol-pred5)^2))##17.10953

plot(testdata$pol,pred4,xlab='actual value',ylab='predicted value',main='predicted value vs actual value')
abline(coef = c(0,1))

residuals.lm<-(testdata$pol-pred4)

qqPlot(residuals.lm,main='LM:residual plot')


##regularization
#ridge

X<-as.matrix(features[,-c(700)])
Y<-features$pol
ridge.mod<-glmnet(X,Y,alpha=0)
plot(ridge.mod,label=T)
ridgetrain<-sample(1:nrow(X),round(0.8*nrow(X)))

cv.out.ridge<-cv.glmnet(X[ridgetrain,],Y[ridgetrain],alpha=0)
plot(cv.out.ridge,main='ridge lamda plot')
bestlamridge<-cv.out.ridge$lambda.min
bestlamridge##4.40515
ridge.predcoef<-predict(cv.out.ridge,s=bestlamridge,type="coefficients")
ridge.predresp<-predict(cv.out.ridge,s=bestlamridge,newx=X[-ridgetrain,],type='response')
ridge_test_RMSE<-sqrt(mean((ridge.predresp-Y[-ridgetrain])^2))##5.972263
residuals.ridge<-(Y[-ridgetrain]-ridge.predresp)
qqPlot(residuals.ridge,main='Ridge LM:residual plot')

coefs = coef(cv.out.ridge)[,1]
coefs = sort(abs(coefs), decreasing = F)
coefs

####lasso
lasso.mod=glmnet(X,Y,alpha=1)
plot(lasso.mod)
lassotrain<-sample(1:nrow(X),round(0.8*nrow(X)))
cv.out.lasso<-cv.glmnet(X[lassotrain,],Y[lassotrain],alpha=1)
plot(cv.out.lasso,main='lasso lamda plot')
bestlamlasso<-cv.out.lasso$lambda.min
bestlamlasso##0.006900391
lasso.predcoef<-predict(cv.out.lasso,s=bestlamlasso,type="coefficients")
lasso.predresp<-predict(cv.out.lasso,s=bestlamlasso,newx=X[-lassotrain,],type='response')
lasso_test_RMSE<-sqrt(mean((lasso.predresp-Y[-lassotrain])^2))###5.915029
residuals.lasso<-(Y[-lassotrain]-lasso.predresp)
qqPlot(residuals.lasso,main='Lasso LM:residual plot')

coefs_la = coef(cv.out.lasso)[,1]
coefs_la = sort(abs(coefs_la), decreasing = F)
coefs_la


##tree model
#single tree

# train=sample(1:nrow(features),round(0.80*nrow(features)))
# test=-train
# testdata=features [test,]
# traindata=features [train,]
tree <- rpart(pol~., data = traindata, method = "anova")
plot(tree,uniform=T,compress=T,main='Tree')
text(tree,use.n=T,all=T,cex=.5)
tree_pred<-predict(tree, data=testdata, type='vector')
tree.RMSE<-sqrt(mean((testdata$pol-tree_pred)^2))##63.70738
residuals.tree<-c()
residuals.tree<-(testdata$pol-tree_pred)
qqPlot(residuals.tree,main='tree:residual plot (c)')



rf.fit<-randomForest(pol~., data=traindata, n.tree=500,mtry=233)
rf_y_hat<-predict(rf.fit,newdata =testdata,type='response')
rf_test_RMSE<-sqrt(mean((testdata$pol-rf_y_hat)^2))##8.678462
varImpPlot(rf.fit)
residuals.rf<-c()
residuals.rf<-(testdata$pol-rf_y_hat)
qqPlot(residuals.rf,main='random forest:residual plot')
partialPlot(rf.fit,traindata,x.var='HyWi_B.p.',main='random forest denpendence plot')



bag.fit<-randomForest(pol~., data=traindata, n.tree=500,mtry=699)
bag_y_hat<-predict(bag.fit,newdata =testdata,type='response')
bag_test_RMSE<-sqrt(mean((testdata$pol-bag_y_hat)^2))##8.889834
varImpPlot(bag.fit)
residuals.bag<-c()
residuals.bag<-(testdata$pol-bag_y_hat)
qqPlot(residuals.bag,main='bagging:residual plot')
partialPlot(bag.fit,traindata,x.var='HyWi_B.p.',main='bagging denpendence plot')










