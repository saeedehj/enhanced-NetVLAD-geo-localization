import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict


def re_ranking_cluster_based(eval_ds, predictions, distances, model_name, approach):
    for query_index, pred in enumerate(predictions):
        utm_list = []
        indx_list = []
        for p in pred:
            path = eval_ds.database_paths[p]
            utm_x = float(path.split("@")[1])
            utm_y = float(path.split("@")[2])
            utm = (utm_x,utm_y)
            utm_list.append(utm)
            indx_list.append(p)
        utm_list= np.array(utm_list)
        distance = np.linalg.norm(utm_list - utm_list[:,None], axis=-1)
        df_distance = pd.DataFrame(distance, columns= indx_list, index= indx_list)
      
        if model_name == 'Agglomorative':
            model = AgglomerativeClustering(affinity='precomputed', n_clusters=5, linkage='complete').fit(df_distance )
            y_db = model.labels_
        elif model_name == 'DBSCAN':
            model =  DBSCAN(eps=5, min_samples=2, metric='precomputed')  # using "precomputed" as recommended by @Anony-Mousse
            y_db = model.fit_predict(df_distance)

        d = defaultdict(list)
        for i, j in zip(df_distance.columns, y_db):
            d[j].append(i)
      
        if approach == 'approach2' or approach =='approach3' or approach =='approach4':
            dist_dic= defaultdict()
            d_list= []
            for i in d.keys():
                for j in d[i]:
                    row = predictions[query_index]
                    index_column = np.argwhere(row == j)[0][0]
                    dis_feat = distances[query_index][index_column]
                    dist_dic[j] = dis_feat
                sort_dict = dict(sorted(dist_dic.items(), key=lambda item: item[1]))
                key_list = list(sort_dict.keys())
                d_list = np.concatenate((d_list, key_list), axis=0)
                d[i] = d_list 
                dist_dic.clear()

        tag_count = defaultdict()
        for k in d.keys():
            tag_count[k] = len(d[k])
            sorted_tag_count =  dict(sorted(tag_count.items(), key=lambda x: x[1], reverse=True))
        
        sorted_list = np.array([])
        if approach == 'approach1' or approach =='approach2':
            for k in sorted_tag_count.keys():
                element_0 = np.array(d[k])
                sorted_list = np.concatenate((sorted_list, np.array(element_0).reshape(-1)), axis=0)
            pred = f_remove_dup(sorted_list)
        elif approach == 'approach3':
            for k in sorted_tag_count.keys():
                element_0 = np.array(d[k])[0]
                sorted_list = np.concatenate((sorted_list, np.array(element_0).reshape(-1)), axis=0)
            for k in sorted_tag_count.keys():
                new_list = np.delete(np.array(d[k]), 0)
                sorted_list = np.concatenate((sorted_list, new_list), axis=0)
            pred = f_remove_dup(sorted_list)
        elif approach == 'approach4':
            for k in sorted_tag_count.keys():
                element_0 = np.array(d[k])[0]
                sorted_list = np.concatenate((sorted_list, np.array(element_0).reshape(-1)), axis=0)
            dist2= defaultdict()
            key_list2 = []
            for i in sorted_list:
                row = predictions[query_index]
                index_column2 = np.argwhere(row == i)[0][0]
                dist_feat2= distances[query_index][index_column2]
                dist2[i] = dist_feat2
                sort_dict2 = dict(sorted(dist2.items(), key=lambda item: item[1]))
                key_list2 = list(sort_dict2.keys())
            sorted_list1= []
            sorted_list1 = np.concatenate((sorted_list1 , key_list2), axis=0)
            for k in sorted_tag_count.keys():
                new_list = np.delete(np.array(d[k]), 0)
                sorted_list1 = np.concatenate((sorted_list1 , new_list), axis=0)
            pred = f_remove_dup(sorted_list1)
        
        predictions[query_index]= pred

    return predictions

def re_ranking_distance_based(eval_ds, predictions, distances, approach):
    for query_index, pred in enumerate(predictions):
        utm_list = []
        indx_list = []
        for p in pred:
            path = eval_ds.database_paths[p]
            utm_x = float(path.split("@")[1])
            utm_y = float(path.split("@")[2])
            utm = (utm_x,utm_y)
            utm_list.append(utm)
            indx_list.append(p)
        utm_list= np.array(utm_list)
        distance = np.linalg.norm(utm_list - utm_list[:,None], axis=-1)
        df_distance = pd.DataFrame(distance, columns= indx_list, index= indx_list)
      
        ind = list()
        for i in df_distance.columns:
            for j in df_distance.columns:
                if df_distance.loc[i][j] < 5: 
                    tuple= (i,j)
                    ind.append(tuple)
        unique_values = set([list[0] for list in ind])
        group_list = [[list for list in ind if list[0] == value] for value in unique_values]  
        sort = sorted(group_list, key= len, reverse=True)
        
        if approach == 'approach1':
            pre= []
            for i in range(0, len(sort)):
                for j in range(0, len(sort[i])):
                    s= sort[i][j][1]
                    pre.append(s)
            pred = f_remove_dup(pre)
        elif approach == 'approach2':
            dist_dic = defaultdict()
            d_list = []
            key_list= []
            for i in range(0, len(sort)):
                for j in range(0, len(sort[i])):
                    row = predictions[query_index]
                    index_column = np.argwhere(row == sort[i][j][1])[0][0]
                    dis_feat = distances[query_index][index_column]
                    dist_dic[sort[i][j][1]] = dis_feat
                sort_dict = dict(sorted(dist_dic.items(), key=lambda item: item[1]))
                key_list = list(sort_dict.keys())
                d_list = np.concatenate((d_list, key_list), axis=0)
                dist_dic.clear()
            pred = f_remove_dup(d_list)
        elif approach == 'approach3':
            dist_dic = defaultdict()
            d_list = []
            key_list= []
            first= []
            others= []
            for i in range(0, len(sort)):
                for j in range(0, len(sort[i])):
                    row = predictions[query_index]
                    index_column = np.argwhere(row == sort[i][j][1])[0][0]
                    dis_feat = distances[query_index][index_column]
                    dist_dic[sort[i][j][1]] = dis_feat
                sort_dict = dict(sorted(dist_dic.items(), key=lambda item: item[1]))
                key_list = list(sort_dict.keys())
                others = np.concatenate((others, key_list[1:]), axis=0)
                first.append(key_list[0])
                dist_dic.clear()
            d_list = np.concatenate((first, others), axis=0)
            pred = f_remove_dup(d_list)
        elif approach == 'approach4':
            dist_dic = defaultdict()
            dist2 = defaultdict()
            d_list = []
            key_list= []
            key_list2= []
            first= []
            others= []
            for i in range(0, len(sort)):
                for j in range(0, len(sort[i])):
                    row = predictions[query_index]
                    index_column = np.argwhere(row == sort[i][j][1])[0][0]
                    dis_feat = distances[query_index][index_column]
                    dist_dic[sort[i][j][1]] = dis_feat
                sort_dict = dict(sorted(dist_dic.items(), key=lambda item: item[1]))
                key_list = list(sort_dict.keys())
                others = np.concatenate((others, key_list[1:]), axis=0)
                first.append(key_list[0])
                for i in first:
                    row = predictions[query_index]
                    index_column2 = np.argwhere(row == i)[0][0]
                    dist_feat2= distances[query_index][index_column2]
                    dist2[i] = dist_feat2
                    sort_dict2 = dict(sorted(dist2.items(), key=lambda item: item[1]))
                    key_list2 = list(sort_dict2.keys())
                dist_dic.clear()
            d_list = np.concatenate((key_list2 , others), axis=0)  
            pred = f_remove_dup(d_list)  
        
    predictions[query_index]= pred
    return predictions

def f_remove_dup(input): 
    seen = set()
    return [x for x in input if x not in seen and not seen.add(x)]

