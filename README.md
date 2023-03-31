# ML Bootcamp

This Repository Contains Notebooks Containing Linear Regression,Polynomial Regression,Logistic Regression,Neural Network,K-nearest Neighbours.

## LinearRegression.ipynb :-

Use modelname = LinearRegression(learning_rate,iter) to initialize the model,where iter is the number of iterations for training.  
Use modelname.fit(X,Y) to fit the model using X against Y.   
Use modelname.predict(X) to make predicitons.   
Use modelname.Showmetrics(X,Y) to show the scoring metrics rmse,mae,mse,r2score when tested on X,with true values Y.   
Contains inbuilt Z-score normalized.  

## PolynomialRegression.ipynb :-

Use modelname = PolynomialRegression(deg,learning_rate=0.001,epochs=10000,lambda1 = 0, lambda2 = 0.02,tol = 0.001,reg_alpha=0) the hyperparameter values can be changed as per user's convinience.   

Contains Inbuilt Elastic Regularisation,which can be used by setting reg_alpha between 0 to 1.   
 Use modelname.fit(X,Y,show_plt=True,show_cst = True,plt_typ='normal') to train the model using X,Y 
 and user can specify whether they want to output the costs at each 1000 iterations,or to show the iteration vs cost plot.   
 
 Use modelname.pred(X) to get the predictions made by the model.  
 Use modelname.show_metrics(X,Y) to show the scoring metrics obtained when the model is tested on X,with Y as true values.  
 Use modelname.tune_hyp_gridsearch(X,Y,X_cv,Y_cv) to perform grid search of hyperparameter using performance on X_cv,Y_cv as a scoring metric.  
 Use modelname.rand_search(X,Y,X_cv,Y_cv,deg) to perform random search of hyperparameter using performance on X_cv,Y_cv as a scoring metric.  
 
 
## Logistic Regression:-
 
 Use model = logisticregression(typ='softmax',learning_rate=0.001,epochs=1000,base=0.5,reg_lambda=0) to initialize the model ,user can specify type of logistic regression,softmax or onevsall,specify the hyperparameters while initializing the model.   
 use model.fit(X,Y,n_classes=10) to train the model using X,Y and user can specify the no of classes   
 use model.pred(X) to make predictions   
 use model.accuracy(X,Y) to get the accuracy of the model.   
 ## L-layer Neural Network:- 
 
 Use model = NNeuralNetwork(layer_sizes,activation) ,user has to input the size of each layer in an list format,and specify the activation function.(Supported activations are sigmoid,ReLU,Leaky ReLU,tanh)
 
 Use model.train(X,Y,X_cv,Y_cv,learning_rate=0.0001,epochs=16,batch_size = 32, beta1=0.9, beta2 = 0.999,epsilon = 1e-8,show_epochs=True) to train the model against X,Y and use X_cv,Y_cv for validation accuracy. user can change the hyperparameters above to their need.
 
 Use model.predict(X) to make predictions
 
 Use model.confusion_matrix(X,Y_test,Y_pred) to get confusion matrix
 
 Use accuracy(X,Y,Y_pred) to get the accuracy
 
