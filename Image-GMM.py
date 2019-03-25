#!/usr/bin/env python
# coding: utf-8

# In[9]:


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


# In[15]:


get_ipython().run_cell_magic('time', '', "max_clusters = 30\nclusters = []\naccuracy = []\nnn_score = []\n_sil_score = []\n_sse_score = []\n_mi_score = []\n_bic_score = []\nfor k in range(2,max_clusters):\n    gmm = GMM(n_components=k, random_state=0).fit(data_train)\n    cluster_labels = gmm.predict(data_train)\n    y = label_train \n    assert (y.shape == cluster_labels.shape)\n    pred = np.empty_like(y)\n    for label in set(cluster_labels):\n        mask = cluster_labels == label\n        sub = y[mask]\n        target = Counter(sub).most_common(1)[0][0]\n        pred[mask] = target\n    #    assert max(pred) == max(Y)\n    #    assert min(pred) == min(Y)\n    this_accuracy = acc(y, pred)\n    print(this_accuracy)\n    clusters.append(k)\n    accuracy.append(this_accuracy)\n    \n    \n    this_sil_score = sil_score(data_train, cluster_labels)\n    print (this_sil_score)\n    _sil_score.append(this_sil_score)\n    this_sse_score = gmm.score(data_train)\n    print (this_sse_score)\n    _sse_score.append(this_sse_score)\n    this_mi_score = ami(y, cluster_labels)\n    print (this_mi_score)\n    _mi_score.append(this_mi_score)\n    this_bic_score = gmm.bic(data_train)\n    print (this_bic_score)\n    _bic_score.append(this_bic_score)\n    \n    \n \n    \n        clf_kmean = MLPClassifier(solver='lbfgs', alpha=1e-06, hidden_layer_sizes=(9), random_state=0, activation ='relu')\n#     GridSearchCV(MLPClassifier(), parameters, n_jobs=5, cv=3)\n    clf_kmean.fit(data_train, pred)\n#     clf_kmean_model = clf_kmean.best_estimator_\n#     print (clf_kmean.best_score_, clf_kmean.best_params_)\n    this_nn_score = f1_score(label_test, clf_kmean.predict(data_test), average='macro')\n    nn_score.append(this_nn_score)\n    print (this_nn_score)\n\n\n    ")


# In[16]:


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




