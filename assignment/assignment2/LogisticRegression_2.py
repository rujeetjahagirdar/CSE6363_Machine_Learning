import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(style="darkgrid")
from sklearn.model_selection import train_test_split
class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = {}
        self.bias = {}
        self.val_losses =[]
        self.train_losses=[]
        self.prediction=[]
        self.posteriors=[]
        self.unique_classes=[]

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, lr=0.01,class_balance_split=False,debug=False):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        if debug: print("inside fit")
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        #X.insert(0,'dummy_bias_feature',np.ones(X.shape[0]))  # adding dummy feature with value 1
        X=np.column_stack([np.ones(X.shape[0]),X])
        self.unique_classes=sorted(list(np.unique(y)))
        if debug : print("classes ",self.unique_classes)
#         self.val_losses =[]
#         self.train_losses=[]
#         self.prediction=[]
        np.random.seed(46)
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1],X.shape[1])) # 2 features and 1 bias term
            #self.bias['bias_'+str(cls)] = np.random.uniform(-1, 1, size=(1))
        print("initial weights ",self.weights)
        #if debug : print("bias ",self.bias)
        # TODO: Implement the training loop.
        #### one hot encoding ########
        y = y.merge(pd.get_dummies(y['species'],prefix='encoded'),on=y.index)
        y.drop(['key_0'],axis=1,inplace=True)
        ########################
        #### train-validation split ########
        train_X, validation_X, train_y, validation_y = train_test_split(X,y,test_size=0.10,random_state=45,stratify=y['species'])
        train_y.drop(['species'],axis=1,inplace=True)
        validation_y.drop(['species'],axis=1,inplace=True)
        if debug : print("training X data shape ",train_X.shape)
        if debug : print("training Y data shape ",train_y.shape)
        if debug : print("validation X data shape ",validation_X.shape)
        if debug : print("validation Y data shape ",validation_y.shape)
        #############################
        current_patience = [0,0,0]
        prev_val_mse_loss = [float('inf'),float('inf'),float('inf')]
        prev_weights=np.ones(shape=(3,3))
        ### 100 steps loop ####
        for n in range(self.max_epochs):
            #### for loop batch processing ###
            if(np.equal(current_patience,[patience,patience,patience]).all()):
                print("!!!!!!!! Max. patience reached !!!!!!!!!!")
                self.weights = prev_weights
                break
            for i in range(0,train_X.shape[0],batch_size):
                if debug: print("Step = {}, batch= {}".format(n+1,i+1))
                train_X_batch_data = train_X[i:i+batch_size]
                train_Y_batch_data = train_y.to_numpy()[i:i+batch_size]
                if debug : print("train_X_batch shape",train_X_batch_data.shape)
                #if debug : print("train_X_batch",train_X_batch_data)
                if debug : print("train_y_batch shape",train_Y_batch_data.shape)
                if debug : print("weights shape",self.weights.shape)
                #for cls in self.unique_classes:
                self.prediction = train_X_batch_data @ self.weights 
                if debug : print("Prediction shape", self.prediction.shape)
                if debug : print("Prediction ", self.prediction)
                #for cls in self.unique_classes:
                num = np.exp(self.prediction)
                self.posteriors = (num/(np.sum(num,axis=1, keepdims=True)))
                #self.posteriors = (np.exp(self.prediction))/(np.sum(np.exp(self.prediction),axis=0, keepdims=True))
                if debug :print("############################################ Sum of posteriers {}".format(np.sum(self.posteriors,axis=1)))
                if debug : print("Posterior shape", self.posteriors.shape)
                if debug : print("Posterior ",self.posteriors)
                if debug : print("Train_Y_Batch shape",train_Y_batch_data.shape)
                #if debug : print("Train_Y_Batch ",train_Y_batch_data)
                error = self.posteriors - train_Y_batch_data
                #if debug : print('error= ',error)
                if debug : print("Error shape ",error.shape)
                #if debug : print("Error ",error)
                if debug : print("Batch L1 Loss (absolute): ",np.absolute(error).mean(axis=0))
                gradient = (error.T @ train_X_batch_data) + (regularization * self.weights)
                if debug : print('gradient= ',gradient)
                self.weights = self.weights - (lr * gradient.T)
            print("######### End of batch processing #########")
            trn_prediction = train_X @ self.weights
            trn_posteriar = (np.exp(trn_prediction))/(np.sum(np.exp(trn_prediction),axis=1, keepdims=True))
            trn_error = trn_posteriar - train_y.to_numpy() # 150x3
            trn_mse_loss = np.absolute(trn_error).mean(axis=0)  # 1x3
            self.train_losses.append(trn_mse_loss)
            if debug : print('### train losses ',self.train_losses)
            val_prediction = validation_X @ self.weights
            val_posteriar = (np.exp(val_prediction))/(np.sum(np.exp(val_prediction),axis=1, keepdims=True))
            val_error = val_posteriar - validation_y.to_numpy()  # 150x3
            val_mse_loss = np.absolute(val_error).mean(axis=0)  # 1x3  [_,_,_]
            self.val_losses.append(val_mse_loss)
            if debug : print('### validation losses ',self.val_losses)
            print("before Current Patience\n",current_patience)
            print("before prev_val_mse_loss\n",prev_val_mse_loss) # 1x3
            print("before val_mse_loss\n",val_mse_loss)  # 1x3
            print("before prev_weights\n",prev_weights)  # 3x3
            for cls in self.unique_classes:
                if(current_patience[cls] <= patience):
                    if(prev_val_mse_loss[cls] > val_mse_loss[cls]):
                        current_patience[cls]= 0 
                        prev_weights[:,cls]=[1,1,1]  # [[pre_weigh_cls1],[pre_weigh_cls2],[pre_weigh_cls3]]
                    else:
                        prev_weights[:,cls]=self.weights[:,cls]
                        current_patience[cls] = current_patience[cls] + 1
                    prev_val_mse_loss[cls] = val_mse_loss[cls]
                    if debug : print("------Validation Mean Squared Error: {} for step {}".format(val_mse_loss,n+1))
            print("Current Patience\n",current_patience)
            print("prev_val_mse_loss\n",prev_val_mse_loss)
            print("val_mse_loss\n",val_mse_loss)
            print("prev_weights\n",prev_weights)
        #print(val_losses)
        print("Final Weights-------> ",self.weights)
            #sns.lineplot(y=val_losses,x=range(len(val_losses)))
    #         sns.lineplot(data=[self.val_losses[cls],self.train_losses[cls]]).set(xlabel='Step no.',ylabel='MSE',xlim=(0,None))
    #         plt.legend(labels=["Validation MSE","Train MSE"])
    #         plt.savefig('result.jpg')
    #         plt.show()

    def predict(self, X,debug=False):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
        # TODO: Implement the prediction function.
        if debug: print("\n\nInside predict")
        #X.insert(0,'dummy_bias_feature',np.ones(X.shape[0]))  # adding dummy feature with value 1
        X=np.column_stack([np.ones(X.shape[0]),X])
        #X=pd.concat([pd.Series(1.0),X])
        prediction = X @ self.weights
        if debug: print("prediction shape",prediction.shape)
        num = np.exp(prediction)
        posteriar = num / np.sum(num, axis=1, keepdims=True)
        if debug: print("posteriar shape",posteriar.shape)
        if debug :print("############################################ Sum of posteriers {}".format(np.sum(posteriar,axis=1)))
        ans = posteriar.max(axis=1)
        if debug: print("ans shape ",ans.shape)
        #ans = prediction.max(axis=1)
        #print(ans)
        return(np.array(ans))

    def score(self, X, y,debug=False):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        predictions = {}
        X.insert(0,'dummy_bias_feature',np.ones(X.shape[0]))  # adding dummy feature with value 1
        # TODO: Implement the scoring function.
        if debug: print("\nInside socre")
        def get_prediction(row):
            ans=[]
            for cls in self.unique_classes:
                prediction = row @ self.weights['weights_'+str(cls)]
                ans.append(np.exp(prediction)/(1+ np.exp(prediction)))
            return ([max(ans),self.unique_classes[ans.index(max(ans))]])
        #if debug: print("Weights used: ",self.weights)
        X.apply(lambda row: print(get_prediction(row)),axis=1)
        error = prediction - y.values
        mse_loss = np.absolute(error).mean()
        if debug: print("\nScore methode MSE= ",mse_loss)
        return (mse_loss)