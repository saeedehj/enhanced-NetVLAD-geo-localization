
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def test_efficient_ram_usage(args, eval_ds, model, test_method="hard_resize"):
    """This function gives the same output as test(), but uses much less RAM.
    This can be useful when testing with large descriptors (e.g. NetVLAD) on large datasets (e.g. San Francisco).
    Obviously it is slower than test(), and can't be used with PCA.
    """
    
    model = model.eval()
    if test_method == 'nearest_crop' or test_method == "maj_voting":
        distances = np.empty([eval_ds.queries_num * 5, eval_ds.database_num], dtype=np.float32)
    else:
        distances = np.empty([eval_ds.queries_num, eval_ds.database_num], dtype=np.float32)

    with torch.no_grad():
        if test_method == 'nearest_crop' or test_method == 'maj_voting':
            queries_features = np.ones((eval_ds.queries_num * 5, args.features_dim), dtype="float32")
        else:
            queries_features = np.ones((eval_ds.queries_num, args.features_dim), dtype="float32")
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            if test_method == "nearest_crop" or test_method == 'maj_voting':
                start_idx = (indices[0] - eval_ds.database_num) * 5
                end_idx = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                queries_features[indices, :] = features.cpu().numpy()
            else:
                queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()

        queries_features = torch.tensor(queries_features).type(torch.float32).cuda()
        
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            inputs = inputs.to(args.device)
            features = model(inputs)
            for pn, (index, pred_feature) in enumerate(zip(indices, features)):
                distances[:, index] = ((queries_features-pred_feature)**2).sum(1).cpu().numpy()
        del features, queries_features, pred_feature
        
    predictions = distances.argsort(axis=1)[:, :max(args.recall_values)]
    
    if test_method == 'nearest_crop':
        distances = np.array([distances[row, index] for row, index in enumerate(predictions)])
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each
    elif test_method == 'maj_voting':
        distances = np.array([distances[row, index] for row, index in enumerate(predictions)])
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query
    del distances
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def test(args, eval_ds, model, test_method="hard_resize", pca=None):
    """Compute features of the given dataset and compute the recalls."""
    
    assert test_method in ["hard_resize", "single_query", "central_crop", "five_crops",
                            "nearest_crop", "maj_voting"], f"test_method can't be {test_method}"
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        eval_ds.test_method = "hard_resize"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        if test_method == "nearest_crop" or test_method == 'maj_voting':
            all_features = np.empty((5 * eval_ds.queries_num + eval_ds.database_num, args.features_dim), dtype="float32")
        else:
            all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

        for inputs, indices in tqdm(database_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            if test_method == "five_crops" or test_method == "nearest_crop" or test_method == 'maj_voting':
                inputs = torch.cat(tuple(inputs))  # shape = 5*bs x 3 x 480 x 480
            features = model(inputs.to(args.device))
            if test_method == "five_crops":  # Compute mean along the 5 crops
                features = torch.stack(torch.split(features, 5)).mean(1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)
            
            if test_method == "nearest_crop" or test_method == 'maj_voting':  # store the features of all 5 crops
                start_idx = eval_ds.database_num + (indices[0] - eval_ds.database_num) * 5
                end_idx   = start_idx + indices.shape[0] * 5
                indices = np.arange(start_idx, end_idx)
                all_features[indices, :] = features
            else:
                all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    if test_method == 'nearest_crop':
        distances = np.reshape(distances, (eval_ds.queries_num, 20 * 5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20 * 5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20]  # keep only the closer 20 predictions for each query
    elif test_method == 'maj_voting':
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distances in-place
            top_n_voting('top1', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top5', predictions[q], distances[q], args.majority_weight)
            top_n_voting('top10', predictions[q], distances[q], args.majority_weight)

            # flatten dist and preds from 5, 20 -> 20*5
            # and then proceed as usual to keep only first 20
            dists = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dists)
            preds = preds[sort_idx]
            # remove duplicated predictions, i.e. keep only the closest ones
            _, unique_idx = np.unique(preds, return_index=True)
            # unique_idx is sorted based on the unique values, sort it again
            # here the row corresponding to the first crop is used as a
            # 'buffer' for each query, and in the end the dimension
            # relative to crops is eliminated
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20]  # keep only the closer 20 predictions for each query

    for query_index, pred in enumerate(predictions):
      utm_list = []
      indx_list = np.array([])
      for p in pred:
        path = eval_ds.database_paths[p]
        utm_x, utm_y = float(path.split("@")[1], path.split("@")[2])
        utm_list.append((utm_x, utm_y))
        indx_list = np.concatenate((indx_list , p), axis=0)
      df_distance = pd.DataFrame(np.linalg.norm(utm_list - utm_list[:,None], axis=-1), columns= indx_list, index= indx_list)
      if args.cluster_type == "DBSCAN":
         model = DBSCAN(eps=5, min_samples=2, metric='precomputed')  
         y = model.fit_predict(df_distance)
      if args.cluster_type == "agglomorative":
         model = AgglomerativeClustering(affinity='precomputed', n_clusters=5, linkage='complete').fit(df_distance )
         y = model.labels_
      #if args.cluster_type == "proposed":
      if args.approach == "approach1":
        cluster_dict = defaultdict(list)
        for i, j in zip(df_distance.columns, y):
            cluster_dict[j].append(i)
      ############################
      if args.approach == "approach2":
        dist_dic= defaultdict()
        d_list= []
        for i in cluster_dict.keys():
            for j in cluster_dict[i]:
              row = predictions[query_index]
              index_column = np.argwhere(row == j)[0][0]
              dis_feat = distances[query_index][index_column]
              dist_dic[j] = dis_feat
            sort_dict = dict(sorted(dist_dic.items(), key=lambda item: item[1]))
            key_list = list(sort_dict.keys())
            d_list = np.concatenate((d_list, key_list), axis=0)
            cluster_dict[i] = d_list 
            dist_dic.clear()
      #################################
      len_cluster = defaultdict()
      for k in cluster_dict.keys():
          len_cluster[k] = len(cluster_dict[k])
          sorted_cluster =  dict(sorted(len_cluster.items(), key=lambda x: x[1], reverse=True))
      sorted_list = np.array([])
      if args.approach == "approach3":
          sorted_list = np.array([])
          for k in sorted_cluster.keys():
             element_0 = np.array(cluster_dict[k])[0]
             sorted_list = np.concatenate((sorted_list, np.array(element_0).reshape(-1)), axis=0)
          for k in sorted_cluster.keys():
             new_list = np.delete(np.array(cluster_dict[k]), 0)
             sorted_list = np.concatenate((sorted_list, new_list), axis=0)
      if args.approach == "approach4":
          sorted_list = np.array([])
          for k in sorted_cluster.keys():
              element_0 = np.array(cluster_dict[k])[0]
              sorted_list = np.concatenate((sorted_list, np.array(element_0).reshape(-1)), axis=0)
          dist2= defaultdict()
          key_list2 = []
          for i in sorted_list:
             index_column2 = np.argwhere(predictions[query_index] == i)[0][0]
             dist_feat2= distances[query_index][index_column2]
             dist2[i] = dist_feat2
             sort_dict2 = dict(sorted(dist2.items(), key=lambda item: item[1]))
             key_list2 = list(sort_dict2.keys())
      sorted_list1= []
      sorted_list = np.concatenate((sorted_list1 , key_list2), axis=0)
      for k in sorted_cluster.keys():
          new_list = np.delete(np.array(d[k]), 0)
          sorted_list1 = np.concatenate((sorted_list , new_list), axis=0)
      else:
          for k in sorted_cluster.keys():
             elements = np.array(cluster_dict[k])
             sorted_list = np.concatenate((sorted_list, np.array(elements).reshape(-1)), axis=0)
      def f(seq): 
          seen = set()
          return [x for x in seq if x not in seen and not seen.add(x)]
      pred = f(sorted_list)
      predictions[query_index]= pred

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str


def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n