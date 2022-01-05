# -*- coding:utf-8 -*-
# J.W
# May 2021 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from kmeans_select import kmeans_select
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='data_expression_median.csv', type = str, help = 'input file')
parser.add_argument('--cluster', default = 6, type = int, help = 'k-means')
parser.add_argument('--feature', default = 500, type = int, help = 'the number of importance feature')
parser.add_argument('--output', default = 'select_feature.csv', type = str, help = 'output file')
opt = parser.parse_args()

def kmeans_select(kmeans_labels,y):

    kmeans_labels2 = []
    y_labels2 = []
    for i in kmeans_labels:
        if i not in kmeans_labels2:
            kmeans_labels2.append(i)
    for i in y:
        if i not in y_labels2:
            y_labels2.append(i)

    df_labels = pd.DataFrame(columns=kmeans_labels2,index=y_labels2)
    df_labels.loc[:, :] = 0
    for i in range(len(y)):
        val = df_labels[kmeans_labels[i]][y[i]]
        df_labels[kmeans_labels[i]][y[i]] = val + 1

    km_argmax = {}
    for row in df_labels.index:
        s1 = df_labels.loc[row,:]

        km_argmax[row]=(s1[s1 == s1.max()].index[0],s1.max())
###############################################################

    new_km = km_argmax.copy()
    selected_klabel = []
    for key1,val1 in km_argmax.items():
        selected_klabel.extend([val1[0]])
        for key2,val2 in km_argmax.items():
            if key1 != key2:
                if (val1[0] == val2[0]) and (val1[1] > val2[1]):
                    new_km[key2] = (-1,val2[1])
    selected_klabel = list(set(selected_klabel))
    for key,val in new_km.items():
        if val[0] == -1:
            remain_label = kmeans_labels2.copy()
            for label_tmp in selected_klabel:
                remain_label.remove(label_tmp)
            ss = df_labels.loc[key, remain_label]
            klabel = ss[ss == ss.max()].index[0]
            new_km[key] = (klabel, ss.max())
            selected_klabel.append(klabel)
###############################################################
    new_label = []
    for i in range(len(kmeans_labels)):
        if kmeans_labels[i] in selected_klabel:
            if new_km[y[i]][0]==kmeans_labels[i]:
                new_label.append(kmeans_labels[i])  
            else:
                new_label.append('delete')   
        else:
            new_label.append(kmeans_labels[i])   
    return new_label


def RF_feature_reduction_by_importance(feature_head, x, y, cnt):
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    count = 1
    while(x.shape[1]>cnt):
    # for i in range(k):
        forest = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        forest.fit(x, y)
        score = forest.score(x,y)
        print('---',count,'---score:',score)
        count = count + 1
        importances = forest.feature_importances_
        indices = np.argsort(importances) # sort ascend
        importances_greater_index=[]
        for inx in indices:
            if (inx/x.shape[1]) >= 0.1:
                importances_greater_index.append(inx)
        x = x[:, importances_greater_index]
        feature_head = feature_head[importances_greater_index]
    return feature_head, x, y



if __name__=='__main__':
   
    print('loading data...')
    data_df = pd.read_csv(opt.input, header=None,low_memory=False)
    print('loaded...')
    data_df_dropna = data_df.dropna(axis=1, how='any')  # drop all cols that have any NaN values
    # data_df_dropna = data_df_dropna.dropna(axis=0, how='any')  # drop all rows that have any NaN values
    indx_all = data_df_dropna.iloc[1:, 0].values
    x, y = data_df_dropna.iloc[1:, 2:].values, data_df_dropna.iloc[1:, 1].values # 0 col is number 1 col is label 2: cols are data
    feature_head = data_df_dropna.iloc[0, 2:].values
    # y_new = pd.Categorical(y).codes
    remain_feature_cnt = opt.feature
    feature_head_reduced, x_reduced, y_reduced = RF_feature_reduction_by_importance(feature_head, x,y,remain_feature_cnt) # remain the first 99% features every time
    print('features reduce...')

    num_of_cluster = opt.cluster
    kmeans_more = KMeans(n_clusters=num_of_cluster)  # n_clusters:number of cluster
    kmeans_more.fit(x_reduced)
   
    selected_label = kmeans_select(kmeans_more.labels_, y_reduced)
    print('labels select...')
    new_x = []  # reduce x, if label exist, x remains
    new_y = []
    new_indx = []
    for i in range(len(selected_label)):
        if selected_label[i] != 'delete':
            new_x.append(x_reduced[i,:])    # it is 2-D
            new_y.append([selected_label[i]])  # to generate 2-Dimension
            new_indx.append([indx_all[i]])
    #print('Feature Reduced!')
    # print(np.array(new_y).shape)
    # print(np.array(new_x).shape)
    data = np.hstack((np.array(new_y),np.array(new_x)))
    # print((np.array(new_y)).shape)
    data = np.hstack((np.array(new_indx),data))
    feature_head_reduced = feature_head_reduced.tolist()
    feature_head_reduced.insert(0,'label')
    feature_head_reduced.insert(0,'Sample')
    new_data = pd.DataFrame(data.tolist(),columns=feature_head_reduced)
    new_data.to_csv(opt.output, index = False)
