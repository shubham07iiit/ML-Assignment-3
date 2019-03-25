#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import imageio
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt        # to plot any graph
import cv2
import os
import random
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import pathlib
import imageio
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt        # to plot any graph
import cv2
import os
import random
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import adjusted_mutual_info_score as ami, silhouette_score as sil_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import SparseRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.mixture import GaussianMixture as GMM

flowers = glob(os.path.join('Shubham-data/17flowers/jpg', "*.jpg"))
aeroplanes = glob(os.path.join('Shubham-data/airplanes_side', "*.jpg"))

# flowers_sorted = sorted([x for x in flowers])
# print(flowers)
# im_path = flowers_sorted[45]
# im = imageio.imread(str(im_path))
# print(im.shape)
# im_gray = rgb2gray(im)
# im_gray.reshape(-1,3)
# print('New image shape: {}'.format(im_gray.shape))

# plt.imshow(im, cmap='Set3')  # show me the leaf
# plt.show()

# plt.imshow(im_gray, cmap='Set3')  # show me the leaf
# plt.show()

def proc_images(images, label):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """



    x = [] # images as arrays
    y = [] # labels Infiltration or Not_infiltration
    WIDTH = 64
    HEIGHT = 64

    for img in images:

        # Read and resize image
        full_size_image = cv2.imread(img, 0)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))

        # Labels
        y.append(label)



    return x,y



flowers_x, flowers_y = proc_images(flowers, 1)
aeroplanes_x, aeroplanes_y = proc_images(aeroplanes, 0)
len(flowers_x)

df = pd.DataFrame()
df["labels"]=aeroplanes_y, flowers_y
df["images"]=aeroplanes_x, flowers_x
df.head()



plt.imshow(flowers_x[0], cmap='Set3')

plt.imshow(aeroplanes_x[0], cmap='Set3')



for i in range(len(flowers_x)):
    flowers_x[i] = flowers_x[i].flatten()

for i in range(len(aeroplanes_x)):
    aeroplanes_x[i] = aeroplanes_x[i].flatten()


training = flowers_x + aeroplanes_x
labels = flowers_y + aeroplanes_y
data_train, data_test, label_train, label_test = train_test_split(training, labels, test_size = 0.2, random_state = 50)
label_train = pd.Series((v for v in label_train))
data_train = pd.DataFrame(data_train)
# print(data_train)



# dt2 = tree.DecisionTreeClassifier(random_state=1, max_depth=4)
# dt2.fit(training, labels)
# dt2_score_train = dt2.score(data_train, label_train)
# print("Training score: ",dt2_score_train)
# dt2_score_test = dt2.score(data_test, label_test)
# print("Testing score: ",dt2_score_test)


# # Let's generate the decision tree for depth = 6
# # Create a feature vector
# #features = training.columns.tolist()

# # Uncomment below to generate the digraph Tree.
# #tree.export_graphviz(dt2, out_file=dot_data, feature_names=features)


# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200), random_state=1)
# clf.fit(data_train, label_train)
# print(clf.score(data_train, label_train))
# print(clf.score(data_test, label_test))
# print(clf.predict(data_test))


# In[3]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.metrics.pairwise import pairwise_distances\ndef pairwise_dist_corr(x1, x2):\n    assert x1.shape[0] == x2.shape[0]\n\n    d1 = pairwise_distances(x1)\n    d2 = pairwise_distances(x2)\n    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]\n\n\nmax_components = 100\ncomponents = []\naccuracy = []\nnn_score = []\n_sil_score = []\n_sse_score = []\n_mi_score = []\n_bic_score = []\n_corr = []\nfor k in range(2,max_components):\n    rp = SparseRandomProjection(n_components=k)\n    new_data_train = rp.fit_transform(data_train)\n    \n    this_corr = pairwise_dist_corr(new_data_train, data_train)\n    print(this_corr)\n    _corr.append(this_corr)\n    \n    clf_rp = MLPClassifier(solver='lbfgs', alpha=1e-06, hidden_layer_sizes=(9), random_state=0, activation ='relu')\n    clf_rp.fit(new_data_train, label_train)\n#     clf_rp_model = clf_rp.best_estimator_\n#     print (clf_rp.best_score_, clf_rp.best_params_)\n    new_data_test = rp.fit_transform(data_test)\n    this_nn_score = f1_score(label_test, clf_rp.predict(new_data_test), average='macro')\n    components.append(k)\n    nn_score.append(this_nn_score)\n    print (this_nn_score)\n\n\n    ")


# In[14]:


max_value = max(nn_score)
max_index = nn_score.index(max_value)
print(max_index)
plt.figure()
plt.title("ImageData - Neural Network Accuracy curve with RP")
plt.xlabel("No Of Components")
plt.ylabel("NN Accuracy")
plt.grid()
# plt.xticks(list(range(1,101)))
plt.plot(components, nn_score, 'o-', label="NN Acuracy")

plt.figure()
plt.title("ImageData - Corelations Coeff with components in RP")
plt.xlabel("No Of Components")
plt.ylabel("Correlation Coeff")
plt.grid()
# plt.xticks(list(range(1,101)))
plt.plot(components, _corr, 'o-', label="NN Acuracy")


# In[15]:


rp = SparseRandomProjection(n_components=max_index)
rp.fit(data_train)
print(data_train.shape)
new_data_train = rp.fit_transform(data_train)
print(new_data_train.shape)
new_data_test = rp.fit_transform(data_test)
print(new_data_test.shape)
# pca.explained_variance_
# do_kmeans_clustering(new_data_train, label_train, new_data_test, label_test)


# In[16]:


get_ipython().run_cell_magic('time', '', "def do_gmm_clustering(data_train, label_train, data_test, label_test):\n    max_clusters = 30\n    clusters = []\n    accuracy = []\n    nn_score = []\n    _sil_score = []\n    _sse_score = []\n    _mi_score = []\n    _bic_score = []\n    for k in range(2,max_clusters):\n        gmm = GMM(n_components=k, random_state=0).fit(data_train)\n        cluster_labels = gmm.predict(data_train)\n        y = label_train \n        assert (y.shape == cluster_labels.shape)\n        pred = np.empty_like(y)\n        for label in set(cluster_labels):\n            mask = cluster_labels == label\n            sub = y[mask]\n            target = Counter(sub).most_common(1)[0][0]\n            pred[mask] = target\n        #    assert max(pred) == max(Y)\n        #    assert min(pred) == min(Y)\n        this_accuracy = acc(y, pred)\n        print(this_accuracy)\n        clusters.append(k)\n        accuracy.append(this_accuracy)\n\n\n        this_sil_score = sil_score(data_train, cluster_labels)\n        print (this_sil_score)\n        _sil_score.append(this_sil_score)\n        this_sse_score = gmm.score(data_train)\n        print (this_sse_score)\n        _sse_score.append(this_sse_score)\n        this_mi_score = ami(y, cluster_labels)\n        print (this_mi_score)\n        _mi_score.append(this_mi_score)\n        this_bic_score = gmm.bic(data_train)\n        print (this_bic_score)\n        _bic_score.append(this_bic_score)\n\n\n\n        parameters = {'solver': ['lbfgs'], 'alpha':10.0 ** -np.arange(1, 5), 'hidden_layer_sizes':np.arange(1, 10), 'random_state':[0]}\n        clf_kmean = GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=5)\n        clf_kmean.fit(data_train, pred)\n        clf_kmean_model = clf_kmean.best_estimator_\n        print (clf_kmean.best_score_, clf_kmean.best_params_)\n        this_nn_score = f1_score(label_test, clf_kmean.predict(data_test), average='macro')\n        nn_score.append(this_nn_score)\n        print (this_nn_score)\n        \n    return clusters, accuracy, _sil_score, _mi_score, _sse_score, _bic_score, nn_score\n\n\n    ")


# In[17]:


get_ipython().run_cell_magic('time', '', "\ndef do_kmeans_clustering(data_train, label_train, data_test, label_test):\n    max_clusters = 30\n    clusters = []\n    accuracy = []\n    nn_score = []\n    _sil_score = []\n    _sse_score = []\n    _mi_score = []\n    for k in range(2,max_clusters):\n        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_train)\n        cluster_labels = kmeans.predict(data_train)\n        y = label_train \n        assert (y.shape == cluster_labels.shape)\n        pred = np.empty_like(y)\n        for label in set(cluster_labels):\n            mask = cluster_labels == label\n            sub = y[mask]\n            target = Counter(sub).most_common(1)[0][0]\n            pred[mask] = target\n        #    assert max(pred) == max(Y)\n        #    assert min(pred) == min(Y)\n        this_accuracy = acc(y, pred)\n        print(this_accuracy)\n        clusters.append(k)\n        accuracy.append(this_accuracy)\n\n\n        this_sil_score = sil_score(data_train, cluster_labels)\n        print (this_sil_score)\n        _sil_score.append(this_sil_score)\n        this_sse_score = kmeans.score(data_train)\n        print (this_sse_score)\n        _sse_score.append(this_sse_score)\n        this_mi_score = ami(y, cluster_labels)\n        print (this_mi_score)\n        _mi_score.append(this_mi_score)\n\n\n\n        clf_kmean = MLPClassifier(solver='lbfgs', alpha=1e-06, hidden_layer_sizes=(9), random_state=0, activation ='relu')\n        clf_kmean.fit(data_train, pred)\n#         clf_kmean_model = clf_kmean.best_estimator_\n#         print (clf_kmean.best_score_, clf_kmean.best_params_)\n        this_nn_score = f1_score(label_test, clf_kmean.predict(data_test), average='macro')\n        nn_score.append(this_nn_score)\n        print (this_nn_score)\n    \n    return clusters, accuracy, _sil_score, _mi_score, _sse_score, nn_score\n\n    ")


# In[18]:


kmeans_clusters, kmeans_accuracy, kmeans_sil_score, kmeans_mi_score, kmeans_sse_score, kmeans_nn_score = do_kmeans_clustering(new_data_train, label_train, new_data_test, label_test)


# In[19]:


gmm_clusters, gmm_accuracy, gmm_sil_score, gmm_mi_score, gmm_sse_score, gmm_bic_score, gmm_nn_score = do_gmm_clustering(new_data_train, label_train, new_data_test, label_test)


# In[20]:


plt.figure()
plt.xticks(list(range(1,30)))
plt.title("ImageData RP - Kmeans + GMM Accuracy curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("Accuracy")
plt.grid()
plt.plot(kmeans_clusters, kmeans_accuracy, 'o-', label="Kmeans")
plt.plot(gmm_clusters, gmm_accuracy, 'o-', label="GMM")
plt.legend(loc='lower right')

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
plt.title("ImageData RP - Kmeans + GMM Neural Network Accuracy curve with number of clusters")
plt.xlabel("No Of Clusters")
plt.ylabel("NN Accuracy")
plt.grid()
plt.plot(kmeans_clusters, kmeans_nn_score, 'o-', label='Kmeans')
plt.plot(gmm_clusters, gmm_nn_score, 'o-', label='GMM')
plt.legend(loc='lower right')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




