# script 
library(dplyr)
library(tidyr)
library(stringr) 
library(lubridate)
library(xgboost)
library(ggplot2)
library(Metrics)
library(fastDummies)


# bringin' in data 
datos_horas=read.csv('datoss.csv')

datos_pred=read.csv('datoss_pred.csv')


datos_horas$Fecha_Contratacion=as.Date(datos_horas$Fecha_Contratacion)
datos_pred$Fecha_Contratacion=as.Date(datos_pred$Fecha_Contratacion)

#datos_horas$Fecha_Contratacion=NULL

#datos_horas$kwh=gsub(',','',datos_horas$kwh)
#datos_horas$kwh=as.integer(datos_horas$kwh)

#sum(is.na(datos_horas$kwh))

#datos_horas=datos_horas[!is.na(datos_horas$kwh),]

# create variables for stuff
datos_horas$wday=wday(datos_horas$Fecha_Contratacion,label=TRUE)
datos_pred$wday=wday(datos_pred$Fecha_Contratacion,label=TRUE)

datos_horas$mes=month(datos_horas$Fecha_Contratacion,label=TRUE)
datos_pred$mes=month(datos_pred$Fecha_Contratacion,label=TRUE)

datos_horas$dia=mday(datos_horas$Fecha_Contratacion)
datos_pred$dia=mday(datos_pred$Fecha_Contratacion)

dummies=dummy_cols(datos_horas,select_columns = c('wday','mes','dia'))

dummies_pred=dummy_cols(datos_pred,select_columns = c('wday','mes','dia'))


# predicting model 
dataxgb=dummies
dataxgb$fecha=NULL
#dataxgb$hora=NULL
dataxgb$mes=NULL
dataxgb$wday=NULL
dataxgb$dia=NULL

dataxgb_pred=dummies_pred
dataxgb_pred$fecha=NULL
#dataxgb_pred$hora=NULL
dataxgb_pred$mes=NULL
dataxgb_pred$wday=NULL
dataxgb_pred$dia=NULL


#dataxgb=datos_horas





##################  
# setting everything for xgboost model 
# Define train and test datasets
N=90
traindata<-dataxgb[c(1:(nrow(dataxgb)-N)),]
testdata<-dataxgb[c( (nrow(dataxgb)-N+1):nrow(dataxgb) ),]

# Generate sample
muestra<-sample(nrow(dataxgb),size =nrow(traindata),replace = FALSE)
watch= dataxgb[muestra,]
watch1=data.matrix(watch%>%select(-Monto_Colocado))
watchtarget= watch$Monto_Colocado

# Watchlist setup
xgval <-  xgb.DMatrix(data = watch1, label= watchtarget)

# Using data.matrix we create the matrix to train with xgboost algorithm
dtrain_matrix<-data.matrix(traindata%>%select(-Monto_Colocado))

# Label for training dataset
trainY<-traindata$Monto_Colocado

# Create the input to the algorithm with xg.DMatrix this is in the xgboost library
dtrainxgb <- xgb.DMatrix(data =dtrain_matrix ,label =trainY, missing = NA)

watchlist1 <- list(val=xgval, train=dtrainxgb)


##### Important, these are the parameters to optimize the xgboost algorithm
params<- list(booster = "gbtree", 
              metric='mae',
              max_depth=25,  
              eta=.02,
              subsample = .8,
              colsample_bytree = .8,
              gamma = 12,
              seed = 8,
              eval_metric = "mae",
              objective = "reg:linear"
)

###### Train the algorithm with the data and the parameters selected
fitxgb <- xgb.train(params = params,
                    data = dtrainxgb,
                    prediction=TRUE,
                    nrounds = 3000,
                    early_stopping_rounds = 250,
                    print_every_n = 100,
                    watchlist = watchlist1
)






# 
# # ##### Important, these are the parameters to optimize the xgboost algorithm
#  params<- list(booster = "dart", 
#                metric='rmse',
#                max_depth=65,  
#                eta=.03,
#                subsample = .7,
#                colsample_bytree = .8,
#                gamma = 3,
#                seed = 8,
#                eval_metric = "rmse",
#                objective = "reg:linear"
#  )
# # 
# # ###### Train the algorithm with the data and the parameters selected
#  fitxgb <- xgb.train(params = params,
#                      data = dtrainxgb,
#                      prediction=TRUE,
#                      nrounds = 1700,
#                      early_stopping_rounds = 50,
#                      print_every_n = 100,
#                      watchlist = watchlist1
#  )




# Coerce the data frame to data matrix to test with the data test
matriz_test<-data.matrix(testdata%>%select(-Monto_Colocado))

# Correct values 
actual=testdata$Monto_Colocado

# Performance monitoring
test_pred <- predict(fitxgb, matriz_test)
train_pred = predict(fitxgb,dtrain_matrix)




df.sup=data.frame(cbind(actual,test_pred))
df.sup$x=c(1:nrow(df.sup))
# Graphing artifacts
g <- ggplot(df.sup, aes(x))
g <- g + geom_line(aes(y=test_pred), colour="red")
g <- g + geom_line(aes(y=actual), colour="blue")
g



#metrics

#evs
sum(test_pred*test_pred)/sum(actual*actual)

sum(train_pred*train_pred)/sum(trainY*trainY)

#mae
mae(actual,test_pred)

mae(trainY,train_pred)

#rmse
rmse(actual,test_pred)

rmse(trainY,train_pred)


##########
df=rbind(actual,test_pred)

df=t(df)

df=as.data.frame(df)

write.csv(df,'predicciones_test.csv')

########

##################  
# setting everything for xgboost model 
# Define train and test datasets
traindata<-dataxgb[c(1:(nrow(dataxgb))),]

# Generate sample
muestra<-sample(nrow(dataxgb),size =nrow(traindata),replace = FALSE)
watch= dataxgb[muestra,]
watch1=data.matrix(watch%>%select(-Monto_Colocado))
watchtarget= watch$Monto_Colocado

# Watchlist setup
xgval <-  xgb.DMatrix(data = watch1, label= watchtarget)

# Using data.matrix we create the matrix to train with xgboost algorithm
dtrain_matrix<-data.matrix(traindata%>%select(-Monto_Colocado))

# Label for training dataset
trainY<-traindata$Monto_Colocado

# Create the input to the algorithm with xg.DMatrix this is in the xgboost library
dtrainxgb <- xgb.DMatrix(data =dtrain_matrix ,label =trainY, missing = NA)

watchlist1 <- list(val=xgval, train=dtrainxgb)


##### Important, these are the parameters to optimize the xgboost algorithm
params<- list(booster = "gbtree", 
              metric='mae',
              max_depth=25,  
              eta=.02,
              subsample = .8,
              colsample_bytree = .8,
              gamma = 12,
              seed = 8,
              eval_metric = "mae",
              objective = "reg:linear"
)

###### Train the algorithm with the data and the parameters selected
fitxgb <- xgb.train(params = params,
                    data = dtrainxgb,
                    prediction=TRUE,
                    nrounds = 3000,
                    early_stopping_rounds = 250,
                    print_every_n = 100,
                    watchlist = watchlist1
)

prediction_matrix<-data.matrix(dataxgb_pred)

mat_aux=prediction_matrix[,55:57]

mat_aux2=prediction_matrix[,1:27]

mat_aux4=prediction_matrix[,28:54]

mat_aux3=as.matrix(prediction_matrix[,58])

prediction_matrix=cbind(mat_aux2,mat_aux,mat_aux4,mat_aux3)

colnames(prediction_matrix)[58]='dia_31'

prediction = predict(fitxgb,prediction_matrix)

plot(prediction,type='l', col='red')
#####
 

df2=c(prediction)

df2=as.data.frame(df2)

colnames(df2)='prediction'

write.csv(df2,'submit_los_ajolotes.csv')




