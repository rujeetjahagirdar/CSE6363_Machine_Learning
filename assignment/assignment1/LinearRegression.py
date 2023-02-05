import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from sklearn.model_selection import train_test_split
class LinearRegression:
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
        self.weights = None
        self.bias = None
        self.val_losses =[]
        self.train_losses=[]

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, class_balance_split=False,debug=False):
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
        print("inside fit")
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1],))
        self.bias = np.random.uniform(-1, 1, size=(1))
        if debug : print("weights ",self.weights.shape)
        if debug : print("bias ",self.bias.shape)
        # TODO: Implement the training loop.
        #### train-validation split ########
        train_X, validation_X, train_y, validation_y = train_test_split(X,y,test_size=0.10,random_state=45,shuffle=True,stratify=None)
        if debug : print("training X data shape ",train_X.shape)
        if debug : print("training Y data shape ",train_y.shape)
        if debug : print("validation X data shape ",validation_X.shape)
        if debug : print("training Y data shape ",validation_y.shape)
############# Uncomment this block if using "stratify" for IRIS dataset   creating data with dependent and independent variables ####
#         independet_vars = ['sepal_length','sepal_width','petal_length']
#         dependent_vars = ['petal_width']
#         train_y = train_X[dependent_vars].values
#         train_X = train_X[independet_vars]
#         validation_y = validation_X[dependent_vars].values
#         validation_X = validation_X[independet_vars]
#         train_X = np.hstack((np.ones((train_X.shape[0], 1)), train_X.values))
#         validation_X = np.hstack((np.ones((validation_X.shape[0], 1)), validation_X.values))
#################################################################################################################################
        current_patience = 0
        prev_val_mse_loss = float('inf')
#         val_losses =[]
#         train_losses=[]
        prev_weights=[]
        ### 100 steps loop ####
        for n in range(self.max_epochs):
            #### for loop batch processing ###
            if(current_patience >= patience):
                print("!!!!!!!! Max. patience reached !!!!!!!!!!")
                #if debug: print("Previous Weights----> ",prev_weights)
                self.weights = prev_weights[0]
                break
            for i in range(0,train_X.shape[0],batch_size):
                if debug: print("Step = {}, batch= {}".format(n+1,i+1))
                train_X_batch_data = train_X[i:i+batch_size]
                train_Y_batch_data = train_y[i:i+batch_size]
                if debug : print("train_X_batch",train_X_batch_data.shape)
                if debug : print("train_y_batch",train_Y_batch_data.shape)
                #print("Adding column of 1s at 1st position in X data.....")
                #train_X_batch_data = np.hstack((np.ones((train_X_batch_data.shape[0], 1)), train_X_batch_data.values))
                #print("train_X_batch",train_X_batch_data.shape)
                prediction = train_X_batch_data.values @ self.weights
                if debug : print("Prediction shape", prediction.shape)
                #if debug : print("Prediction = ",prediction)
                #if debug : print("train_y_batch ",train_Y_batch_data)
                #print("train_Y_batch_data shape= ",train_Y_batch_data.flatten().shape)
#                 error = prediction - train_Y_batch_data.flatten()
                error = prediction - train_Y_batch_data.values
                if debug : print("Error shape ",error.shape)
                print("Batch Mean Squared Error: ",(error**2).mean())
                gradient = error @ train_X_batch_data
                #print("Gradient ",gradient)
                self.weights = self.weights - (0.001 * gradient)
                #print("New Weights ",self.weights)
                #break
            #print("Final Weights-------> ",self.weights)
            #print("Predicting on validation dataset------- for step {}".format(n+1))
            #validation_X = np.hstack((np.ones((validation_X.shape[0], 1)), validation_X.values))
            #print("validation shape", validation_X.shape)
            trn_prediction = train_X @ self.weights
#             val_error = val_prediction - validation_y.flatten()
            trn_error = trn_prediction - train_y
            trn_mse_loss = (trn_error**2).mean()
            self.train_losses.append(trn_mse_loss)
            
            val_prediction = validation_X @ self.weights
#             val_error = val_prediction - validation_y.flatten()
            val_error = val_prediction - validation_y
            val_mse_loss = (val_error**2).mean()
            self.val_losses.append(val_mse_loss)
            if(prev_val_mse_loss > val_mse_loss):
                current_patience= 0 
            else:
                prev_weights.append(self.weights)
                current_patience = current_patience + 1
            prev_val_mse_loss = val_mse_loss
            print("------Validation Mean Squared Error: {} for step {}".format(val_mse_loss,n+1))
        #print(val_losses)
        print("Final Weights-------> ",self.weights)
        #sns.lineplot(y=val_losses,x=range(len(val_losses)))
#         sns.lineplot(data=[self.val_losses,self.train_losses]).set(xlabel='Step no.',ylabel='MSE',xlim=(0,None))
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
#         X = np.hstack((np.ones((X.shape[0], 1)), X.values))
        if debug: print("\nInside predict")
        if debug: print("Weights used: ",self.weights)
        prediction = X @ self.weights
        if debug: print("Predict methode ",prediction)
        return(prediction)

    def score(self, X, y,debug=False):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
#         X = np.hstack((np.ones((X.shape[0], 1)), X.values))
        if debug: print("\nInside socre")
        if debug: print("Weights used: ",self.weights)
        prediction = X @ self.weights
        #print(y.values.flatten())
#         error = prediction - y.values.flatten()
        error = prediction - y.values
        mse_loss = (error**2).mean()
        if debug: print("\nScore methode MSE= ",mse_loss)
        return (mse_loss)