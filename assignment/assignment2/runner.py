import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from LogisticRegression import LogisticRegression
#from LogisticRegression_2 import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from tabulate import tabulate
data = pd.read_csv('iris.csv')
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions

le = LabelEncoder()
train_x, test_x, train_y, test_y = train_test_split(data[['sepal_length','sepal_width','petal_length','petal_width']],data[['species']],test_size=0.10,random_state=46,stratify=data['species'])
#train_x = data[['sepal_length','sepal_width','petal_length','petal_width']]
train_x = train_x[['sepal_length','sepal_width']].to_numpy()
train_y['species']=le.fit_transform(train_y['species'])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
#train_y = train_y.merge(pd.get_dummies(train_y['species'],prefix='encoded'),on=train_y.index)
#train_y.drop(['key_0'],axis=1,inplace=True)
l=LogisticRegression()
l.fit(train_x,train_y,debug=False,max_epochs=100,regularization=0,lr=0.01,batch_size=32)
#print("Validation losses\n",l.val_losses)
print(l.predict(train_x,debug=True))
valid_losses = np.array(l.val_losses)
trn_losses = np.array(l.train_losses)
print('valid_losses shape ',valid_losses.shape)
print('trn_losses shape ',trn_losses.shape)
for cls in range(0,3):
	sns.lineplot(data=[valid_losses[:,cls],trn_losses[:,cls]]).set(xlabel='Step no.',ylabel='L1 Loss',xlim=(0,None))
	plt.legend(labels=["Validation L1 Loss","Train L1 Loss"])
	plt.title("Loss for {}".format(cls))
	plt.savefig('result_{}.jpg'.format(cls))
	plt.show()

ax = plt.subplot(1,1,1)
fig = plot_decision_regions(X=train_x, y=train_y['species'].values, clf=l)
plt.title('Decision Boundries')
plt.savefig('result.jpg')
plt.show()

##############3


