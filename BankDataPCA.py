#!/usr/bin/env python
# coding: utf-8

# In[99]:


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
from sklearn.decomposition import PCA
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


# In[105]:


pca = PCA(n_components=27)
pca.fit(data_train)
print(data_train.shape)
pca.transform(data_train).shape
pca.explained_variance_


# In[101]:


import pandas as pd
plt.figure()
plt.grid()
plt.xticks(list(range(1,28)))
plt.title("BankData - PCA covariance with number of components")
plt.xlabel("No Of Components")
plt.ylabel("Covariance")
# tmp = pd.Series(data=pca.explained_variance_, index=range(1, min(pca.explained_variance_.shape[0], 500) + 1))
# print(tmp.shape)
plt.plot(list(range(1,28)), pca.explained_variance_, '-o')


# In[120]:


get_ipython().run_cell_magic('time', '', "def do_gmm_clustering(data_train, label_train, data_test, label_test):\n    max_clusters = 30\n    clusters = []\n    accuracy = []\n    nn_score = []\n    _sil_score = []\n    _sse_score = []\n    _mi_score = []\n    _bic_score = []\n    for k in range(2,max_clusters):\n        gmm = GMM(n_components=k, random_state=0).fit(data_train)\n        cluster_labels = gmm.predict(data_train)\n        y = label_train \n        assert (y.shape == cluster_labels.shape)\n        pred = np.empty_like(y)\n        for label in set(cluster_labels):\n            mask = cluster_labels == label\n            sub = y[mask]\n            target = Counter(sub).most_common(1)[0][0]\n            pred[mask] = target\n        #    assert max(pred) == max(Y)\n        #    assert min(pred) == min(Y)\n        this_accuracy = acc(y, pred)\n        print(this_accuracy)\n        clusters.append(k)\n        accuracy.append(this_accuracy)\n\n\n#         this_sil_score = sil_score(data_train, cluster_labels)\n#         print (this_sil_score)\n#         _sil_score.append(this_sil_score)\n#         this_sse_score = gmm.score(data_train)\n#         print (this_sse_score)\n#         _sse_score.append(this_sse_score)\n#         this_mi_score = ami(y, cluster_labels)\n#         print (this_mi_score)\n#         _mi_score.append(this_mi_score)\n#         this_bic_score = gmm.bic(data_train)\n#         print (this_bic_score)\n#         _bic_score.append(this_bic_score)\n\n\n\n        parameters = {'solver': ['lbfgs'], 'alpha':10.0 ** -np.arange(1, 5), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0]}\n        clf_kmean = GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=5)\n        clf_kmean.fit(data_train, pred)\n        clf_kmean_model = clf_kmean.best_estimator_\n        print(clf_kmean.best_score_, clf_kmean.best_params_)\n        this_nn_score = f1_score(label_test, clf_kmean.predict(data_test), average='macro')\n        nn_score.append(this_nn_score)\n        print(this_nn_score)\n        \n    return clusters, accuracy, nn_score\n\n\n    ")


# In[106]:


get_ipython().run_cell_magic('time', '', "\ndef do_kmeans_clustering(data_train, label_train, data_test, label_test):\n    max_clusters = 30\n    clusters = []\n    accuracy = []\n    nn_score = []\n    _sil_score = []\n    _sse_score = []\n    _mi_score = []\n    for k in range(2,max_clusters):\n        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_train)\n        cluster_labels = kmeans.predict(data_train)\n        y = label_train \n        assert (y.shape == cluster_labels.shape)\n        pred = np.empty_like(y)\n        for label in set(cluster_labels):\n            mask = cluster_labels == label\n            sub = y[mask]\n            target = Counter(sub).most_common(1)[0][0]\n            pred[mask] = target\n        #    assert max(pred) == max(Y)\n        #    assert min(pred) == min(Y)\n        this_accuracy = acc(y, pred)\n        print(this_accuracy)\n        clusters.append(k)\n        accuracy.append(this_accuracy)\n\n\n        this_sil_score = sil_score(data_train, cluster_labels)\n        print (this_sil_score)\n        _sil_score.append(this_sil_score)\n        this_sse_score = kmeans.score(data_train)\n        print (this_sse_score)\n        _sse_score.append(this_sse_score)\n        this_mi_score = ami(y, cluster_labels)\n        print (this_mi_score)\n        _mi_score.append(this_mi_score)\n\n\n\n        parameters = {'solver': ['lbfgs'], 'alpha':10.0 ** -np.arange(1, 5), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0]}\n        clf_kmean = GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=5)\n        clf_kmean.fit(data_train, pred)\n        clf_kmean_model = clf_kmean.best_estimator_\n        print (clf_kmean.best_score_, clf_kmean.best_params_)\n        this_nn_score = f1_score(label_test, clf_kmean.predict(data_test), average='macro')\n        nn_score.append(this_nn_score)\n        print (this_nn_score)\n    \n    return clusters, accuracy, _sil_score, _mi_score, _sse_score, nn_score\n\n    ")


# In[107]:


get_ipython().run_cell_magic('time', '', "max_components = 27\ncomponents = []\naccuracy = []\nnn_score = []\n_sil_score = []\n_sse_score = []\n_mi_score = []\n_bic_score = []\nfor k in range(2,max_components):\n    pca = PCA(n_components=k).fit(data_train)\n    new_data_train = pca.transform(data_train)\n    \n    parameters = {'solver': ['lbfgs'], 'alpha':10.0 ** -np.arange(1, 5), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0]}\n    clf_pca = GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=5)\n    clf_pca.fit(new_data_train, label_train)\n    clf_pca_model = clf_pca.best_estimator_\n    print (clf_pca.best_score_, clf_pca.best_params_)\n    new_data_test = pca.transform(data_test)\n    this_nn_score = f1_score(label_test, clf_pca.predict(new_data_test), average='macro')\n    components.append(k)\n    nn_score.append(this_nn_score)\n    print (this_nn_score)\n\n\n    ")


# In[108]:


plt.figure()
plt.title("BankData - Neural Network Accuracy curve with PCA")
plt.xlabel("No Of Components")
plt.ylabel("NN Accuracy")
plt.grid()
plt.xticks(list(range(1,28)))
plt.plot(components, nn_score, 'o-', label="NN Acuracy")


# In[109]:


pca = PCA(n_components=12)
pca.fit(data_train)
print(data_train.shape)
new_data_train = pca.transform(data_train)
new_data_test = pca.transform(data_test)
pca.explained_variance_
# do_kmeans_clustering(new_data_train, label_train, new_data_test, label_test)


# In[110]:


kmeans_clusters, kmeans_accuracy, kmeans_sil_score, kmeans_mi_score, kmeans_sse_score, kmeans_nn_score = do_kmeans_clustering(new_data_train, label_train, new_data_test, label_test)


# In[95]:


# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - PCA Kmeans Accuracy curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("Accuracy")
# plt.grid()
# plt.plot(clusters, accuracy, 'o-', label="Kmeans Accuracy")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - PCA Kmeans Sil Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("Sil Score")
# plt.grid()
# plt.plot(clusters, _sil_score, 'o-', label="Kmeans Sil Score")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - PCA Kmeans Mi Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("MI Score")
# plt.grid()
# plt.plot(clusters, _mi_score, 'o-', label="Kmeans MI Score")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - PCA Kmeans Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("Score")
# plt.grid()
# plt.plot(clusters, _sse_score, 'o-', label="Kmeans Score")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - PCA Neural Network Accuracy curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("NN Accuracy")
# plt.grid()
# plt.plot(clusters, nn_score, 'o-', label="NN Acuracy")


# In[ ]:


clusters


# In[121]:


gmm_clusters, gmm_accuracy, gmm_nn_score = do_gmm_clustering(new_data_train, label_train, new_data_test, label_test)


# In[128]:


plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData PCA - Kmeans + GMM Accuracy curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(kmeans_clusters, kmeans_accuracy, 'o-', label="Kmeans")
plt.plot(gmm_clusters, gmm_accuracy, 'o-', label="GMM")
plt.legend(loc='center right')

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - GMM Sil Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("Sil Score")
# plt.grid()
# plt.plot(clusters, _sil_score, 'o-', label="Kmeans Sil Score")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - GMM Mi Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("MI Score")
# plt.grid()
# plt.plot(clusters, _mi_score, 'o-', label="Kmeans MI Score")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - GMM Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("Score")
# plt.grid()
# plt.plot(clusters, _sse_score, 'o-', label="Kmeans Score")

# plt.figure()
# plt.xticks(list(range(1,30)))
# plt.title("BankData - GMM BIC Score curve with number of clusters")
# plt.xlabel("No Of Clusters")
# plt.ylabel("BIC Score")
# plt.grid()
# plt.plot(clusters, _bic_score, 'o-', label="Kmeans Score")

plt.figure()
plt.xticks(list(range(1,30)))
plt.title("BankData PCA - Kmeans + GMM Neural Network Accuracy curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("NN Accuracy")
plt.grid()
plt.plot(kmeans_clusters, kmeans_nn_score, 'o-', label="Kmean")
plt.plot(gmm_clusters, gmm_nn_score, 'o-', label="GMM")
plt.legend(loc='center right')


# In[ ]:





# In[ ]:




