#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
import pydotplus
from IPython.display import Image, display
from sklearn.externals.six import StringIO
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score as acc
from collections import Counter
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score as sil_score
from sklearn.mixture import GaussianMixture as GMM
dot_data = StringIO()

bank = pd.read_csv('Shubham-data/bank.csv')
bank.head()

# Check if the data set contains any null values - Nothing found!
bank[bank.isnull().any(axis=1)].count()

g = sns.boxplot(x=bank["age"])

bank_data = bank.copy()

# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data['poutcome'] = bank_data['poutcome'].replace(['other'] , 'unknown')

# Make a copy for parsing
bank_data.poutcome.value_counts()

# Combine similar jobs into categories
bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
bank_data['job'] = bank_data['job'].replace(['services','housemaid'], 'pink-collar')
bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')

bank_data.job.value_counts()
# Combine 'unknown' and 'other' as 'other' isn't really match with either 'success' or 'failure'
bank_data['poutcome'] = bank_data['poutcome'].replace(['other'], 'unknown')
bank_data.poutcome.value_counts()

# Drop 'contact', as every participant has been contacted.
bank_data.drop('contact', axis=1, inplace=True)

# values for "default" : yes/no
bank_data['default_cat'] = bank_data['default'].map({'yes': 1, 'no': 0})
bank_data.drop('default', axis=1, inplace=True)

# values for "housing" : yes/no
bank_data["housing_cat"]=bank_data['housing'].map({'yes':1, 'no':0})
bank_data.drop('housing', axis=1,inplace = True)

# values for "loan" : yes/no
bank_data["loan_cat"] = bank_data['loan'].map({'yes':1, 'no':0})
bank_data.drop('loan', axis=1, inplace=True)

# day  : last contact day of the month
# month: last contact month of year
# Drop 'month' and 'day' as they don't have any intrinsic meaning
bank_data.drop('month', axis=1, inplace=True)
bank_data.drop('day', axis=1, inplace=True)

# values for "deposit" : yes/no
bank_data["deposit_cat"] = bank_data['deposit'].map({'yes':1, 'no':0})
bank_data.drop('deposit', axis=1, inplace=True)

# Map padys=-1 into a large value (10000 is used) to indicate that it is so far in the past that it has no effect
bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000

# Create a new column: recent_pdays
bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1/bank_data.pdays, 1/bank_data.pdays)

# Convert categorical variables to dummies
bank_with_dummies = pd.get_dummies(data=bank_data, columns = ['job', 'marital', 'education', 'poutcome'],                                    prefix = ['job', 'marital', 'education', 'poutcome'])
bank_with_dummies.head()

bank_with_dummies.describe()

bank_with_dummies.plot(kind='scatter', x='age', y='balance');

bankcl = bank_with_dummies
corr = bankcl.corr()
plt.figure(figsize = (10,10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .82})
plt.title('Heatmap of Correlation Matrix')

bank_with_dummies[bank_data.deposit_cat == 1].describe()

data_drop_deposite = bankcl.drop('deposit_cat', 1)
label = bankcl.deposit_cat
# data_drop_deposite
# label
data_train, data_test, label_train, label_test = train_test_split(data_drop_deposite, label, test_size = 0.2, random_state = 50)
# data_test =  (30, 1500, 1000, 4, 10000, 0, 0)


# In[9]:


get_ipython().run_cell_magic('time', '', "max_clusters = 30\nclusters = []\naccuracy = []\nnn_score = []\n_sil_score = []\n_sse_score = []\n_mi_score = []\n_bic_score = []\nfor k in range(2,max_clusters):\n    gmm = GMM(n_components=k, random_state=0).fit(data_train)\n    cluster_labels = gmm.predict(data_train)\n    y = label_train \n    assert (y.shape == cluster_labels.shape)\n    pred = np.empty_like(y)\n    for label in set(cluster_labels):\n        mask = cluster_labels == label\n        sub = y[mask]\n        target = Counter(sub).most_common(1)[0][0]\n        pred[mask] = target\n    #    assert max(pred) == max(Y)\n    #    assert min(pred) == min(Y)\n    this_accuracy = acc(y, pred)\n    print(this_accuracy)\n    clusters.append(k)\n    accuracy.append(this_accuracy)\n    \n    \n    this_sil_score = sil_score(data_train, cluster_labels)\n    print (this_sil_score)\n    _sil_score.append(this_sil_score)\n    this_sse_score = gmm.score(data_train)\n    print (this_sse_score)\n    _sse_score.append(this_sse_score)\n    this_mi_score = ami(y, cluster_labels)\n    print (this_mi_score)\n    _mi_score.append(this_mi_score)\n    this_bic_score = gmm.bic(data_train)\n    print (this_bic_score)\n    _bic_score.append(this_bic_score)\n    \n    \n \n    parameters = {'solver': ['lbfgs'], 'alpha':10.0 ** -np.arange(1, 5), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0]}\n    clf_kmean = GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=5)\n    clf_kmean.fit(data_train, pred)\n    clf_kmean_model = clf_kmean.best_estimator_\n    print (clf_kmean.best_score_, clf_kmean.best_params_)\n    this_nn_score = f1_score(label_test, clf_kmean.predict(data_test), average='macro')\n    nn_score.append(this_nn_score)\n    print (this_nn_score)\n\n\n    ")


# In[14]:


from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covars_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)

plot_gmm(gmm, data_train)


# In[11]:


plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData - GMM Accuracy curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(clusters, accuracy, 'o-', label="Kmeans Accuracy")

plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData - GMM Sil Score curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("Sil Score")
plt.grid()
plt.plot(clusters, _sil_score, 'o-', label="Kmeans Sil Score")

plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData - GMM Mi Score curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("MI Score")
plt.grid()
plt.plot(clusters, _mi_score, 'o-', label="Kmeans MI Score")

plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData - GMM Score curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("Score")
plt.grid()
plt.plot(clusters, _sse_score, 'o-', label="Kmeans Score")

plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData - GMM BIC Score curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("BIC Score")
plt.grid()
plt.plot(clusters, _bic_score, 'o-', label="Kmeans Score")

plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData - Neural Network Accuracy curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("NN Accuracy")
plt.grid()
plt.plot(clusters, nn_score, 'o-', label="NN Acuracy")


# In[ ]:




