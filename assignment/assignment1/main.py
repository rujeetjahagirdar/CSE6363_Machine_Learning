import matplotlib.pyplot as plt
import seaborn as sns
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from tabulate import tabulate
data = pd.read_csv('iris.csv')
####################
# train_x, test_x, train_y, test_y = train_test_split(data[['sepal_length','sepal_width','petal_length','petal_width']],data[['species']],stratify=data[['species']],test_size=0.10)
# l=LinearRegression()
# # print(test_x.head())
# l.fit(train_x[['sepal_length','sepal_width','petal_length','petal_width']],train_y[['species']])
# l.predict(test_x[['sepal_length','sepal_width','petal_length']])
# l.score(test_x[['sepal_length','sepal_width','petal_length']],test_x[['petal_width']])
####################
models={'model1': {'x':['sepal_length','sepal_width','petal_length'],'y':'petal_width'},
'model2': {'x':['sepal_length','sepal_width'],'y':'petal_width'},
'model3': {'x':['petal_length'],'y':'petal_width'},
'model4': {'x':['sepal_length'],'y':'sepal_width'},
'model5': {'x':['sepal_length','petal_length','petal_width'],'y':'sepal_width'}}

results = {'model1':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model2':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model3':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model4':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]},
          'model5':{'validation_mse':[],'train_mse':[],'test_mse':[],'weights':[]}}
# train_x, test_x, train_y, test_y = train_test_split(data[['sepal_length','sepal_width','petal_length']],data['petal_width'],test_size=0.10)
train_x_split, test_x_split, train_y_split, test_y_split = train_test_split(data[['sepal_length','sepal_width','petal_length','petal_width']],data[['species']],stratify=data[['species']],test_size=0.10)
for model in models:
    print("######### {} ##########".format(model))
    train_x, test_x, train_y, test_y = train_x_split[models[model]['x']], test_x_split[models[model]['x']], train_x_split[models[model]['y']], test_x_split[models[model]['y']] 
    l=LinearRegression()
    # print(train_y.head())
    l.fit(train_x,train_y,debug=False)
#     print(l.val_losses)
#     print(l.train_losses)
#     print(l.weights)
    results[model]['validation_mse']=l.val_losses
    results[model]['train_mse']=l.train_losses
    results[model]['weights']=l.weights
    l.predict(test_x,debug=False)
    test_mse = l.score(test_x,test_y,debug=False)
    print("\nScore methode MSE= ",test_mse)
    results[model]['test_mse']=test_mse
    sns.lineplot(data=[l.val_losses,l.train_losses]).set(xlabel='Step no.',ylabel='MSE',xlim=(0,None))
    plt.title(model)
    plt.legend(labels=["Validation MSE","Train MSE"])
    plt.savefig('{}_result.jpg'.format(model))
    plt.show()
print_data = []
for model in results:
    temp=[]
    temp.append(model)
    temp.append(sum(results[model]['validation_mse'])/len(results[model]['validation_mse']))
    temp.append(sum(results[model]['train_mse'])/len(results[model]['validation_mse']))
    temp.append(results[model]['test_mse'])
    print_data.append(temp)
print("\n")
print(tabulate(print_data,headers=['model','mean_validation_mse','mean_train_mse','test_mse']))