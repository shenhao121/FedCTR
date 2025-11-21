import sys
import os

from pathlib import Path


parent = Path(os.path.abspath("")).resolve().parents[0]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import copy
from logging import INFO
from ml.utils.logger import log
import random

from collections import OrderedDict

import numpy as np
import torch

import pandas as pd

from matplotlib import pyplot as plt

from argparse import Namespace
from ml.utils.data_utils import read_data, generate_time_lags, time_to_feature, handle_nans, to_Xy, \
    to_torch_dataset, to_timeseries_rep, assign_statistics, \
    to_train_val, scale_features, get_data_by_area, remove_identifiers, get_exogenous_data_by_area, handle_outliers, split_exogenous_by_indices
from ml.utils.train_utils import train, test

from ml.models.mlp import MLP
from ml.models.rnn import RNN
from ml.models.lstm import LSTM
from ml.models.gru import GRU
from ml.models.cnn import CNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder

from ml.fl.defaults import create_regression_client
from ml.fl.client_proxy import SimpleClientProxy
from ml.fl.server.server import Server
from ml.utils.helpers import accumulate_metric
from ml.utils.model_utils import *  # 从 model_utils 导入



print(f"Script arguments: {args}\n")


device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
print(f"Using {device}")


# Outlier detection specification
if args.outlier_detection is not None:
    outlier_columns = ['meter_reading']
    outlier_kwargs = {"building1": (10, 90), "building2": (10, 90), "building3": (10, 90)}
    args.outlier_columns = outlier_columns
    args.outlier_kwargs = outlier_kwargs


def seed_all():
    # ensure reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all()

def update_scaler_keys(y_scalers, client_keys):
    updated_scalers = {}
    for client_key in client_keys:
        area = client_key.split("_Client_")[0]
        if area in y_scalers:
            updated_scalers[client_key] = y_scalers[area]
    return updated_scalers

def make_preprocessing():
    """Preprocess a given .csv"""

    exogenous_data_train = None  # 先初始化，避免UnboundLocalError
    exogenous_data_val = None    # 先初始化，避免UnboundLocalError

    # read data
    df = read_data(args.data_path)
    # handle nans
    df = handle_nans(train_data=df, constant=args.nan_constant,
                     identifier=args.identifier)
    # split to train/validation
    train_data, val_data = to_train_val(df)
    
    # handle outliers (if specified)
    if args.outlier_detection is not None:
        train_data = handle_outliers(df=train_data, columns=args.outlier_columns,
                                     identifier=args.identifier, kwargs=args.outlier_kwargs)
    
    # get X and y
    X_train, X_val, y_train, y_val = to_Xy(train_data=train_data, val_data=val_data,
                                          targets=args.targets)
    


    #  # 数据增强：加入噪声
    # def add_noise_to_timeseries(data, noise_level=0.01):
    #     noise = np.random.normal(0, noise_level, data.shape)
    #     return data + noise

    # X_train = add_noise_to_timeseries(X_train, noise_level=0.01)
    # X_val = add_noise_to_timeseries(X_val, noise_level=0.01)

    # # 数据增强：滑动窗口采样
    # def sliding_window_sampling(data, window_size=10, step_size=5):
    #     sampled_data = []
    #     for i in range(0, len(data) - window_size + 1, step_size):
    #         sampled_data.append(data[i:i + window_size])
    #     return np.array(sampled_data)

    # X_train = sliding_window_sampling(X_train, window_size=10, step_size=5)
    # y_train = sliding_window_sampling(y_train, window_size=10, step_size=5)







    # scale X
    X_train, X_val, x_scalers = scale_features(train_data=X_train, val_data=X_val,
                                              scaler=args.x_scaler,
                                              per_area=True, # the features are scaled locally
                                              identifier=args.identifier)
    # scale y
    y_train, y_val, y_scalers = scale_features(train_data=y_train, val_data=y_val,
                                              scaler=args.y_scaler, 
                                              per_area=True,
                                              identifier=args.identifier)
    
    # generate time lags
    X_train = generate_time_lags(X_train, args.num_lags)
    X_val = generate_time_lags(X_val, args.num_lags)
    y_train = generate_time_lags(y_train, args.num_lags, is_y=True)
    y_val = generate_time_lags(y_val, args.num_lags, is_y=True)
    


    # 新增：滑窗后重置索引，保证所有DataFrame索引一致
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    if exogenous_data_train is not None:
        exogenous_data_train = exogenous_data_train.reset_index(drop=True)
    if exogenous_data_val is not None:
        exogenous_data_val = exogenous_data_val.reset_index(drop=True)

    print("原始meter_reading样例：", df["meter_reading"].values[:5])
    print("归一化后y_train样例：", y_train[:5])
    

    # get datetime features as exogenous data
    date_time_df_train = time_to_feature(
        X_train, args.use_time_features, identifier=args.identifier
    )
    date_time_df_val = time_to_feature(
        X_val, args.use_time_features, identifier=args.identifier
    )
    
    # get statistics as exogenous data
    stats_df_train = assign_statistics(X_train, args.assign_stats, args.num_lags,
                                       targets=args.targets, identifier=args.identifier)
    stats_df_val = assign_statistics(X_val, args.assign_stats, args.num_lags, 
                                       targets=args.targets, identifier=args.identifier)
    
    # concat the exogenous features (if any) to a single dataframe
    if date_time_df_train is not None or stats_df_train is not None:
        exogenous_data_train = pd.concat([date_time_df_train, stats_df_train], axis=1)
        # remove duplicate columns (if any)
        exogenous_data_train = exogenous_data_train.loc[:, ~exogenous_data_train.columns.duplicated()].copy()
        assert len(exogenous_data_train) == len(X_train) == len(y_train)
    else:
        exogenous_data_train = None
    if date_time_df_val is not None or stats_df_val is not None:
        exogenous_data_val = pd.concat([date_time_df_val, stats_df_val], axis=1)
        exogenous_data_val = exogenous_data_val.loc[:, ~exogenous_data_val.columns.duplicated()].copy()
        assert len(exogenous_data_val) == len(X_val) == len(y_val)
    else:
        exogenous_data_val = None
        
    return X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers


X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers = make_preprocessing()


# 保留一份原始的
original_y_scalers = y_scalers.copy()

print("Original y_scalers keys:", original_y_scalers.keys())

def make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers):
    """Make data ready to be fed into ml algorithms"""

    if X_train[args.identifier].nunique() != 1:
        split_props_dict = {
            "building1": [0.7, 0.1, 0.1, 0.05, 0.05],
            "building2": [0.02, 0.24, 0.55, 0.04, 0.15],
            "building3": [0.01, 0.1, 0.85, 0.02, 0.02]
        }
        area_X_train, area_X_val, area_y_train, area_y_val = {}, {}, {}, {}
        area_train_indices, area_val_indices = {}, {}
        for area in X_train[args.identifier].unique():
            split_props = split_props_dict[area]
            res = get_data_by_area(
                X_train[X_train[args.identifier] == area],
                X_val[X_val[args.identifier] == area],
                y_train[y_train[args.identifier] == area],
                y_val[y_val[args.identifier] == area],
                identifier="building_id",
                clients_per_area=5,
                seed=args.seed,
                split_props=split_props
            )
            tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val, tmp_train_indices, tmp_val_indices = res
            area_X_train.update(tmp_X_train)
            area_X_val.update(tmp_X_val)
            area_y_train.update(tmp_y_train)
            area_y_val.update(tmp_y_val)
            area_train_indices.update(tmp_train_indices)
            area_val_indices.update(tmp_val_indices)
        # 后续外生特征同步切分同原来
        area_exogenous_data_train = split_exogenous_by_indices(exogenous_data_train, area_train_indices, identifier="building_id")
        area_exogenous_data_val = split_exogenous_by_indices(exogenous_data_val, area_val_indices, identifier="building_id")
    else:
        area_X_train, area_X_val, area_y_train, area_y_val = None, None, None, None
        area_exogenous_data_train, area_exogenous_data_val = None, None



    #  if there are more than one specified areas, get the data per area
    # if X_train[args.identifier].nunique() != 1:
    #     # 切分
    #     split_props = [0.01,0.1,0.85,0.02,0.02]  # 客户端0占99.99%，其余共享0.01%  # 你想要的比例，必须加起来等于1

    #     area_X_train, area_X_val, area_y_train, area_y_val, area_train_indices, area_val_indices = get_data_by_area(
    #         X_train, X_val, y_train, y_val, identifier="building_id", clients_per_area=5,seed=args.seed,split_props=split_props
    #     )
    #     # # 外生特征同步切分（推荐用 get_exogenous_data_by_area，保证顺序切分）
    #     # area_exogenous_data_train, area_exogenous_data_val = get_exogenous_data_by_area(
    #     # exogenous_data_train, exogenous_data_val, identifier="building_id", clients_per_area=5
    #     # )       

    #     area_exogenous_data_train = split_exogenous_by_indices(exogenous_data_train, area_train_indices, identifier="building_id")
    #     area_exogenous_data_val = split_exogenous_by_indices(exogenous_data_val, area_val_indices, identifier="building_id")
    #     # # 外生特征同步切分（只用这一步！）
    #     # area_exogenous_data_train = split_exogenous_by_indices(exogenous_data_train, area_train_indices, identifier="building_id")
    #     # area_exogenous_data_val = split_exogenous_by_indices(exogenous_data_val, area_val_indices, identifier="building_id")
    # else:
    #     area_X_train, area_X_val, area_y_train, area_y_val = None, None, None, None
    #     area_exogenous_data_train, area_exogenous_data_val = None, None#

    # transform to np
    if area_X_train is not None:
        for area in area_X_train:
            tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val = remove_identifiers(
                area_X_train[area], area_y_train[area], area_X_val[area], area_y_val[area])
            tmp_X_train, tmp_y_train = tmp_X_train.to_numpy(), tmp_y_train.to_numpy()
            tmp_X_val, tmp_y_val = tmp_X_val.to_numpy(), tmp_y_val.to_numpy()
            area_X_train[area] = tmp_X_train
            area_X_val[area] = tmp_X_val
            area_y_train[area] = tmp_y_train
            area_y_val[area] = tmp_y_val

    # 这里直接用 area_exogenous_data_train/val，不要再 to_numpy
    exogenous_data_train = area_exogenous_data_train
    exogenous_data_val = area_exogenous_data_val

    # remove identifiers from features, targets
    X_train, y_train, X_val, y_val = remove_identifiers(X_train, y_train, X_val, y_val)
    assert len(X_train.columns) == len(X_val.columns)

    num_features = len(X_train.columns) // args.num_lags

    # to timeseries representation
    X_train = to_timeseries_rep(X_train.to_numpy(), num_lags=args.num_lags,
                                            num_features=num_features)
    X_val = to_timeseries_rep(X_val.to_numpy(), num_lags=args.num_lags,
                                          num_features=num_features)

    if area_X_train is not None:
        area_X_train = to_timeseries_rep(area_X_train, num_lags=args.num_lags,
                                                     num_features=num_features)
        area_X_val = to_timeseries_rep(area_X_val, num_lags=args.num_lags,
                                                   num_features=num_features)

    # transform targets to numpy
    y_train, y_val = y_train.to_numpy(), y_val.to_numpy()

    # # 合并所有客户端的 exogenous_data（如果需要 "all" 键）
    # if exogenous_data_train is not None:
    #     exogenous_data_train_combined, exogenous_data_val_combined = [], []
    #     for area in exogenous_data_train:
    #         exogenous_data_train_combined.extend(exogenous_data_train[area])
    #         exogenous_data_val_combined.extend(exogenous_data_val[area])
    #     exogenous_data_train["all"] = np.stack(exogenous_data_train_combined)
    #     exogenous_data_val["all"] = np.stack(exogenous_data_val_combined)




    if exogenous_data_train is not None:
        exogenous_data_train_combined, exogenous_data_val_combined = [], []
        for area in exogenous_data_train:
            arr_train = np.asarray(exogenous_data_train[area])
            arr_val = np.asarray(exogenous_data_val[area])
            if arr_train.ndim == 1:
                arr_train = arr_train.reshape(1, -1)
            if arr_val.ndim == 1:
                arr_val = arr_val.reshape(1, -1)
            exogenous_data_train_combined.append(arr_train)
            exogenous_data_val_combined.append(arr_val)
    exogenous_data_train["all"] = np.vstack(exogenous_data_train_combined)
    exogenous_data_val["all"] = np.vstack(exogenous_data_val_combined)





    # 新增
    client_keys = list(area_X_train.keys())
    y_scalers = update_scaler_keys(y_scalers, client_keys)

    return X_train, X_val, y_train, y_val, area_X_train, area_X_val, area_y_train, area_y_val, exogenous_data_train, exogenous_data_val, y_scalers


# #均分的逻辑
# def make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers):
#     """Make data ready to be fed into ml algorithms"""
#     # if there are more than one specified areas, get the data per area
#     if X_train[args.identifier].nunique() != 1:
#         # 切分
#         area_X_train, area_X_val, area_y_train, area_y_val, area_train_indices, area_val_indices = get_data_by_area(
#             X_train, X_val, y_train, y_val, identifier="District", clients_per_area=5
#         )

#         # 外生特征同步切分
#         area_exogenous_data_train = split_exogenous_by_indices(exogenous_data_train, area_train_indices, identifier="District")
#         area_exogenous_data_val = split_exogenous_by_indices(exogenous_data_val, area_val_indices, identifier="District")
#         # area_X_train, area_X_val, area_y_train, area_y_val = get_data_by_area(X_train, X_val,
#         #                                                                       y_train, y_val, 
#         #                                                                       identifier=args.identifier,
#         #                                                                       clients_per_area=5)
#         # area_X_train, area_X_val, area_y_train, area_y_val = get_data_by_area(X_train, X_val,
#         #                                                                       y_train, y_val, 
#         #                                                                       identifier=args.identifier,
#         #                                                                       clients_per_area=5
#         #                                                                    )
#     else:
#         area_X_train, area_X_val, area_y_train, area_y_val = None, None, None, None

#     # Get the exogenous data per area.
#     if exogenous_data_train is not None:
#         # exogenous_data_train, exogenous_data_val = get_exogenous_data_by_area(exogenous_data_train,
#         #                                                                       exogenous_data_val)
#         exogenous_data_train, exogenous_data_val = get_exogenous_data_by_area(exogenous_data_train,
#                                                                               exogenous_data_val,
#                                                                                identifier=args.identifier, clients_per_area=5)
#     # transform to np
#     if area_X_train is not None:
#         for area in area_X_train:
#             tmp_X_train, tmp_y_train, tmp_X_val, tmp_y_val = remove_identifiers(
#                 area_X_train[area], area_y_train[area], area_X_val[area], area_y_val[area])
#             tmp_X_train, tmp_y_train = tmp_X_train.to_numpy(), tmp_y_train.to_numpy()
#             tmp_X_val, tmp_y_val = tmp_X_val.to_numpy(), tmp_y_val.to_numpy()
#             area_X_train[area] = tmp_X_train
#             area_X_val[area] = tmp_X_val
#             area_y_train[area] = tmp_y_train
#             area_y_val[area] = tmp_y_val
    
#     if exogenous_data_train is not None:
#         for area in exogenous_data_train:
#             exogenous_data_train[area] = exogenous_data_train[area].to_numpy()
#             exogenous_data_val[area] = exogenous_data_val[area].to_numpy()
    
#     # remove identifiers from features, targets
#     X_train, y_train, X_val, y_val = remove_identifiers(X_train, y_train, X_val, y_val)
#     assert len(X_train.columns) == len(X_val.columns)
    
#     num_features = len(X_train.columns) // args.num_lags
    
#     # to timeseries representation
#     X_train = to_timeseries_rep(X_train.to_numpy(), num_lags=args.num_lags,
#                                             num_features=num_features)
#     X_val = to_timeseries_rep(X_val.to_numpy(), num_lags=args.num_lags,
#                                           num_features=num_features)
    
#     if area_X_train is not None:
#         area_X_train = to_timeseries_rep(area_X_train, num_lags=args.num_lags,
#                                                      num_features=num_features)
#         area_X_val = to_timeseries_rep(area_X_val, num_lags=args.num_lags,
#                                                    num_features=num_features)
    
#     # transform targets to numpy
#     y_train, y_val = y_train.to_numpy(), y_val.to_numpy()
    
#     if exogenous_data_train is not None:
#         exogenous_data_train_combined, exogenous_data_val_combined = [], []
#         for area in exogenous_data_train:
#             exogenous_data_train_combined.extend(exogenous_data_train[area])
#             exogenous_data_val_combined.extend(exogenous_data_val[area])
#         exogenous_data_train_combined = np.stack(exogenous_data_train_combined)
#         exogenous_data_val_combined = np.stack(exogenous_data_val_combined)
#         exogenous_data_train["all"] = exogenous_data_train_combined
#         exogenous_data_val["all"] = exogenous_data_val_combined


#     #新增
#     client_keys = list(area_X_train.keys())
#     y_scalers = update_scaler_keys(y_scalers, client_keys)


#     return X_train, X_val, y_train, y_val, area_X_train, area_X_val, area_y_train, area_y_val, exogenous_data_train, exogenous_data_val,y_scalers


X_train, X_val, y_train, y_val, client_X_train, client_X_val, client_y_train, client_y_val, exogenous_data_train, exogenous_data_val,y_scalers = make_postprocessing(X_train, X_val, y_train, y_val, exogenous_data_train, exogenous_data_val, x_scalers, y_scalers)

# exogenous_data_train.keys()

# exogenous_data_train["ElBorn"]
# 打印所有 ElBorn 子客户端的键和数据形状
elborn_clients = [key for key in exogenous_data_train.keys() if "ElBorn" in key]
for client in elborn_clients:
    print(f"Client: {client}, Data Shape: {exogenous_data_train[client].shape}")

# 计算某些统计信息
for client in elborn_clients:
    print(f"Client: {client}, Feature Length: {len(exogenous_data_train[client][0])}")
#
#
def get_input_dims(X_train, exogenous_data_train):
    if args.model_name == "mlp":
        input_dim = X_train.shape[1] * X_train.shape[2]
    else:
        input_dim = X_train.shape[2]

    if exogenous_data_train is not None:
        if len(exogenous_data_train) == 1:
            cid = next(iter(exogenous_data_train.keys()))
            exogenous_dim = exogenous_data_train[cid].shape[1]
        else:
            exogenous_dim = exogenous_data_train["all"].shape[1]
    else:
        exogenous_dim = 0

    return input_dim, exogenous_dim



input_dim, exogenous_dim = get_input_dims(X_train, exogenous_data_train)

print(input_dim, exogenous_dim)

model = get_model(model=args.model_name,
                  input_dim=input_dim,
                  out_dim=y_train.shape[1],
                  lags=args.num_lags,
                  exogenous_dim=exogenous_dim,
                  seed=args.seed)

print(model)

def fit(model, X_train, y_train, X_val, y_val, 
        exogenous_data_train=None, exogenous_data_val=None, 
        idxs=[0], # the indices of our targets in X
        log_per=1,
        client_creation_fn = None, # client specification
        local_train_params=None, # local params
        aggregation_params=None, # aggregation params
        use_carbontracker=True
       ):
    # client creation definition
    if client_creation_fn is None:
        client_creation_fn = create_regression_client
    # local params
    if local_train_params is None:
        local_train_params = {
            "epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
            "criterion": args.criterion, "early_stopping": args.local_early_stopping,
            "patience": args.local_patience, "device": device
        }
    
    train_loaders, val_loaders = [], []
    
    # get data per client
    for client in X_train:
        if client == "all":
            continue
        if exogenous_data_train is not None:
            tmp_exogenous_data_train = exogenous_data_train[client]
            tmp_exogenous_data_val = exogenous_data_val[client]
        else:
            tmp_exogenous_data_train = None
            tmp_exogenous_data_val = None



        print(f"[{client}] X: {X_train[client].shape}, y: {y_train[client].shape}, exogenous: {tmp_exogenous_data_train.shape if tmp_exogenous_data_train is not None else None}")
        num_features = len(X_train[client][0][0])
        
        # to torch loader
        train_loaders.append(
            to_torch_dataset(
                X_train[client], y_train[client],
                num_lags=args.num_lags,
                num_features=num_features,
                exogenous_data=tmp_exogenous_data_train,
                indices=idxs,
                batch_size=args.batch_size,
                shuffle=False
            )
        )
        val_loaders.append(
            to_torch_dataset(
                X_val[client], y_val[client],
                num_lags=args.num_lags,
                exogenous_data=tmp_exogenous_data_val,
                indices=idxs,
                batch_size=args.batch_size,
                shuffle=False
            )
            
        )
        
    # # create clients with their local data
    # cids = [k for k in X_train.keys() if k != "all"]
    # clients = [
    #     client_creation_fn(
    #         cid=cid, # client id
    #         model=model, # the global model
    #         train_loader=train_loader, # the local train loader
    #         test_loader=val_loader, # the local val loader
    #         local_params=local_train_params # local parameters
    #     )
    #     for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders)
    # ]

    cids = [k for k in X_train.keys() if k != "all"]
    clients = [
        client_creation_fn(
            cid=cid,  # 客户端 ID
            model=model,  # 全局模型
            train_loader=train_loader,  # 本地训练数据
            test_loader=val_loader,  # 本地验证数据
            local_params=local_train_params  # 本地训练参数
        )
        for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders)
    ]
    
    # represent clients to server
    client_proxies = [
        SimpleClientProxy(cid, client) for cid, client in zip(cids, clients)
    ]
    
    # represent the server
    server = Server(

        X_train=X_train,  #新增
        exogenous_data_train=exogenous_data_train, #新增
        client_proxies=client_proxies, # the client representations
        aggregation=args.aggregation, # the aggregation algorithm
        aggregation_params=aggregation_params, # aggregation specific params
        local_params_fn=None, # we can change the local params on demand
    )
    # Note that the client manager instance will be initialized automatically. You can define your own client manager.

    # train with FL
    # model_params, history = server.fit(args.fl_rounds, args.fraction, use_carbontracker=use_carbontracker)
    
    global_model, history = server.fit(args.fl_rounds, args.fraction, use_carbontracker=use_carbontracker)

    
    #
    # print("Model parameters:", model_params)
    # print("Type of model_params:", type(model_params))

    # params_dict = zip(model.state_dict().keys(), model_params)
    # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    # model = copy.deepcopy(model)
    # model.load_state_dict(state_dict, strict=True)
    
    return global_model, history


# federated local params
local_train_params = {"epochs": args.epochs, "optimizer": args.optimizer, "lr": args.lr,
                      "criterion": args.criterion, "early_stopping": args.local_early_stopping,
                      "patience": args.local_patience, "device": device
                      }

global_model, history = fit(
    model,
    client_X_train,
    client_y_train, 
    client_X_val, 
    client_y_val, 
    local_train_params=local_train_params,
    exogenous_data_train=exogenous_data_train,
    exogenous_data_val=exogenous_data_val
)



def transform_preds(y_pred_train, y_pred_val):
    if not isinstance(y_pred_train, np.ndarray):
        y_pred_train = y_pred_train.cpu().numpy()
    if not isinstance(y_pred_val, np.ndarray):
        y_pred_val = y_pred_val.cpu().numpy()
    return y_pred_train, y_pred_val

def round_predictions(y_pred_train, y_pred_val, dims):
    # round to closest integer
    if dims is None or len(dims) == 0:
        return y_pred_train, y_pred_val
    for dim in dims:
        y_pred_train[:, dim] = np.rint(y_pred_train[:, dim])
        y_pred_val[:, dim] = np.rint(y_pred_val[:, dim])
    return y_pred_train, y_pred_val

def inverse_transform(y_train, y_val, y_pred_train, y_pred_val,
                     y_scaler=None, 
                     round_preds=False, dims=None):
    y_pred_train, y_pred_val = transform_preds(y_pred_train, y_pred_val)
    
    if y_scaler is not None:
        y_train = y_scaler.inverse_transform(y_train)
        y_val = y_scaler.inverse_transform(y_val)
        y_pred_train = y_scaler.inverse_transform(y_pred_train)
        y_pred_val = y_scaler.inverse_transform(y_pred_val)
    
    # to zeroes
    y_pred_train[y_pred_train < 0.] = 0.
    y_pred_val[y_pred_val < 0.] = 0.
    
    if round_preds:
        y_pred_train, y_pred_val = round_predictions(y_pred_train, y_pred_val, dims)
    
    return y_train, y_val, y_pred_train, y_pred_val


def make_plot(y_true, y_pred, 
              title, 
              feature_names=None, 
              client=None,
              save_dir="./plots"):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(y_pred.shape[1])]
    assert len(feature_names) == y_pred.shape[1]





    print("画图用的y_true[:5]:", y_true[:5])
    print("画图用的y_pred[:5]:", y_pred[:5])

     # 创建保存目录（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for i in range(y_pred.shape[1]):
        plt.figure(figsize=(8, 6))
        plt.ticklabel_format(style='plain')
        plt.plot(y_true[:, i], label="Actual")
        plt.plot(y_pred[:, i], label="Predicted")
        if client is not None:
            plt.title(f"[{client} {title}] {feature_names[i]} prediction")
            save_path = os.path.join(save_dir, f"{client}_{title}_{feature_names[i]}.png")
        else:
            plt.title(f"[{title}] {feature_names[i]} prediction")
            save_path = os.path.join(save_dir, f"{title}_{feature_names[i]}.png")

        plt.legend()
        plt.savefig(save_path)  # 保存图像到文件
        plt.close()


# def inference(
#     model, # the global model
#     client_X_train, # train data per client
#     client_y_train,
#     client_X_val, # val data per client
#     client_y_val,
#     exogenous_data_train, # exogenous data per client
#     exogenous_data_val,
#     y_scalers, # the scaler used to transform the targets
#     idxs=[8, 3, 1, 10, 9],
#     apply_round=True, # round to closest integer
#     round_dimensions=[0, 3, 4], # the dimensions to apply rounding
#     plot=True, # plot predictions
# ):
#     # load per client data to torch
#     train_loaders, val_loaders = [], []
    
#     # get data per client
#     for client in client_X_train:
#         if client == "all":
#             continue
#         assert client in list(y_scalers.keys())
#         if exogenous_data_train is not None:
#             tmp_exogenous_data_train = exogenous_data_train[client]
#             tmp_exogenous_data_val = exogenous_data_val[client]
#         else:
#             tmp_exogenous_data_train = None
#             tmp_exogenous_data_val = None
    
#         num_features = len(client_X_train[client][0][0])
        
#         # to torch loader
#         train_loaders.append(
#             to_torch_dataset(
#                 client_X_train[client], client_y_train[client],
#                 num_lags=args.num_lags,
#                 num_features=num_features,
#                 exogenous_data=tmp_exogenous_data_train,
#                 indices=idxs,
#                 batch_size=1,
#                 shuffle=False
#             )
#         )
#         val_loaders.append(
#             to_torch_dataset(
#                 client_X_val[client], client_y_val[client],
#                 num_lags=args.num_lags,
#                 exogenous_data=tmp_exogenous_data_val,
#                 indices=idxs,
#                 batch_size=1,
#                 shuffle=False
#             )
            
#         )
        
#     # get client ids
#     cids = [k for k in client_X_train.keys() if k != "all"]
        
#     # predict per client using the global model
#     y_preds_train, y_preds_val = dict(), dict()
#     for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders):
#         print(f"Prediction on {cid}")
#         train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
#             model, train_loader, None, device=device
#         )
#         val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
#             model, val_loader, None, device=device
#         )
#         y_preds_train[cid] = y_pred_train
#         y_preds_val[cid] = y_pred_val
    
#     for cid in cids:
#         y_train, y_val = client_y_train[cid], client_y_val[cid]
#         y_pred_train, y_pred_val = y_preds_train[cid], y_preds_val[cid]
        
#         y_scaler = y_scalers[cid]
#         y_train, y_val, y_pred_train, y_pred_val = inverse_transform(
#             y_train, y_val, y_pred_train, y_pred_val,
#             y_scaler, round_preds=apply_round, dims=round_dimensions
#         )
#         train_mse, train_rmse, train_mae, train_r2, train_nrmse, train_res_per_dim = accumulate_metric(
#             y_train, y_pred_train, True, return_all=True
#         )
#         val_mse, val_rmse, val_mae, val_r2, val_nrmse, val_res_per_dim = accumulate_metric(
#             y_val, y_pred_val, True, return_all=True
#         )
        
#         print(f"\nFinal Prediction on {cid} (Inference Stage)")
#         print(f"[Train]: mse: {train_mse}, "
#               f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
#         print(f"[Val]: mse: {val_mse}, "
#               f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}, nrmse: {val_nrmse}\n\n")
        
#         if plot:
#             make_plot(y_train, y_pred_train, title="Train", feature_names=args.targets, client=cid)
#             make_plot(y_val, y_pred_val, title="Val", feature_names=args.targets, client=cid)
# def inference(
#     model,  # the global model
#     client_X_train,  # train data per client
#     client_y_train,
#     client_X_val,  # val data per client
#     client_y_val,
#     exogenous_data_train,  # exogenous data per client
#     exogenous_data_val,
#     original_y_scalers,  # 使用原始的 y_scalers
#     idxs=[0],
#     apply_round=True,  # round to closest integer
#     round_dimensions=[0],  # the dimensions to apply rounding
#     plot=True,  # plot predictions
# ):
#     # load per client data to torch
#     train_loaders, val_loaders = [], []

#     # get data per client
#     for client in client_X_train:
#         if client == "all":
#             continue


#         # area = int(client.split("_Client_")[0])  # 转成int，和y_scalers的key类型一致
#         area = client.split("_Client_")[0]  # 从子客户端键中提取大地点



#            # 在这里加打印，帮助你定位问题
#         print(f"当前 client: {client}, area: {area}")
#         print(f"original_y_scalers.keys(): {list(original_y_scalers.keys())}")

#         assert area in original_y_scalers.keys()  # 检查大地点是否在 original_y_scalers 中
#         if exogenous_data_train is not None:
#             tmp_exogenous_data_train = exogenous_data_train[client]
#             tmp_exogenous_data_val = exogenous_data_val[client]
#         else:
#             tmp_exogenous_data_train = None
#             tmp_exogenous_data_val = None

#         num_features = len(client_X_train[client][0][0])

#         # to torch loader
#         train_loaders.append(
#             to_torch_dataset(
#                 client_X_train[client],
#                 client_y_train[client],
#                 num_lags=args.num_lags,
#                 num_features=num_features,
#                 exogenous_data=tmp_exogenous_data_train,
#                 indices=idxs,
#                 batch_size=1,
#                 shuffle=False,
#             )
#         )
#         val_loaders.append(
#             to_torch_dataset(
#                 client_X_val[client],
#                 client_y_val[client],
#                 num_lags=args.num_lags,
#                 exogenous_data=tmp_exogenous_data_val,
#                 indices=idxs,
#                 batch_size=1,
#                 shuffle=False,
#             )
#         )

#     # get client ids
#     cids = [k for k in client_X_train.keys() if k != "all"]

#     # predict per client using the global model
#     y_preds_train, y_preds_val = dict(), dict()
#     for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders):
#         print(f"Prediction on {cid}")
#         train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
#             model, train_loader, None, device=device
#         )
#         val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
#             model, val_loader, None, device=device
#         )
#         y_preds_train[cid] = y_pred_train
#         y_preds_val[cid] = y_pred_val

#     # 合并子客户端的预测结果到 3 个大的地点
#     merged_y_train, merged_y_val = dict(), dict()
#     train_metrics = []
#     val_metrics = []



#     for area in ["building1", "building2", "building3"]:
#         # 使用原始的 y_scalers
#         y_scaler = original_y_scalers[area]

#         # 合并训练数据
#         merged_y_train[area] = np.concatenate(
#             [y_preds_train[cid].cpu().numpy() for cid in cids if area in cid], axis=0
#         )
#         # 合并验证数据
#         merged_y_val[area] = np.concatenate(
#             [y_preds_val[cid].cpu().numpy() for cid in cids if area in cid], axis=0
#         )

#     # 对每个大的地点进行预测结果处理和绘图
#     for area in ["building1", "building2", "building3"]:
#         y_train = np.concatenate(
#             [client_y_train[cid] for cid in cids if area in cid], axis=0
#         )
#         y_val = np.concatenate(
#             [client_y_val[cid] for cid in cids if area in cid], axis=0
#         )
#         y_pred_train = merged_y_train[area]
#         y_pred_val = merged_y_val[area]



#         # 在inverse_transform前
#         print("Before inverse_transform:")
#         print("y_train[:5]:", y_train[:5])
#         print("y_pred_train[:5]:", y_pred_train[:5])





#         # 使用对应的 scaler 进行逆变换
#         y_train, y_val, y_pred_train, y_pred_val = inverse_transform(
#             y_train,
#             y_val,
#             y_pred_train,
#             y_pred_val,
#             y_scaler,
#             round_preds=apply_round,
#             dims=round_dimensions,
#         )


#                 # 逆归一化后
#         print("After inverse_transform:")
#         print("y_train[:5]:", y_train[:5])
#         print("y_pred_train[:5]:", y_pred_train[:5])






#         # 计算指标
#         train_mse, train_rmse, train_mae, train_r2, train_nrmse, train_res_per_dim = accumulate_metric(
#             y_train, y_pred_train, True, return_all=True
#         )
#         val_mse, val_rmse, val_mae, val_r2, val_nrmse, val_res_per_dim = accumulate_metric(
#             y_val, y_pred_val, True, return_all=True
#         )
#         train_metrics.append([train_mse, train_rmse, train_mae, train_r2, train_nrmse])
#         val_metrics.append([val_mse, val_rmse, val_mae, val_r2, val_nrmse])


#         log(INFO, f"\nFinal Prediction on {area} (Inference Stage)")
#         log(INFO, f"[Train]: mse: {train_mse}, "
#                   f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
#         log(INFO, f"[Val]: mse: {val_mse}, "
#                   f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}, nrmse: {val_nrmse}\n\n")

#         # print(f"\nFinal Prediction on {area} (Inference Stage)")
#         # print(
#         #     f"[Train]: mse: {train_mse}, "
#         #     f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}"
#         # )
#         # print(
#         #     f"[Val]: mse: {val_mse}, "
#         #     f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}, nrmse: {val_nrmse}\n\n"
#         # )

#         if plot:
#             make_plot(
#                 y_train,
#                 y_pred_train,
#                 title="Train",
#                 feature_names=args.targets,
#                 client=area,
#             )
#             make_plot(
#                 y_val,
#                 y_pred_val,
#                 title="Val",
#                 feature_names=args.targets,
#                 client=area,
#             )
#     # 转为numpy方便计算
#     train_metrics = np.array(train_metrics)
#     val_metrics = np.array(val_metrics)

#     # 计算平均
#     avg_train_metrics = train_metrics.mean(axis=0)
#     avg_val_metrics = val_metrics.mean(axis=0)

#     log(INFO, "\n=== 全局平均指标 (三地区平均) ===")
#     log(INFO, f"[Train] mse: {avg_train_metrics[0]}, rmse: {avg_train_metrics[1]}, mae: {avg_train_metrics[2]}, r2: {avg_train_metrics[3]}, nrmse: {avg_train_metrics[4]}")
#     log(INFO, f"[Val]   mse: {avg_val_metrics[0]}, rmse: {avg_val_metrics[1]}, mae: {avg_val_metrics[2]}, r2: {avg_val_metrics[3]}, nrmse: {avg_val_metrics[4]}")
#     # print("\n=== 全局平均指标 (三地区平均) ===")
#     # print(f"[Train] mse: {avg_train_metrics[0]}, rmse: {avg_train_metrics[1]}, mae: {avg_train_metrics[2]}, r2: {avg_train_metrics[3]}, nrmse: {avg_train_metrics[4]}")
#     # print(f"[Val]   mse: {avg_val_metrics[0]}, rmse: {avg_val_metrics[1]}, mae: {avg_val_metrics[2]}, r2: {avg_val_metrics[3]}, nrmse: {avg_val_metrics[4]}")

# def inference(
#     model,  # the global model
#     client_X_train,  # train data per client
#     client_y_train,
#     client_X_val,  # val data per client
#     client_y_val,
#     exogenous_data_train,  # exogenous data per client
#     exogenous_data_val,
#     original_y_scalers,  # 使用原始的 y_scalers
#     idxs=[0],
#     apply_round=False,  # 关闭四舍五入
#     round_dimensions=[0],  # the dimensions to apply rounding
#     plot=True,  # plot predictions
# ):
#     # load per client data to torch
#     train_loaders, val_loaders = [], []

#     # get data per client
#     for client in client_X_train:
#         if client == "all":
#             continue
#         if exogenous_data_train is not None:
#             tmp_exogenous_data_train = exogenous_data_train[client]
#             tmp_exogenous_data_val = exogenous_data_val[client]
#         else:
#             tmp_exogenous_data_train = None
#             tmp_exogenous_data_val = None

#         num_features = len(client_X_train[client][0][0])

#         train_loaders.append(
#             to_torch_dataset(
#                 client_X_train[client],
#                 client_y_train[client],
#                 num_lags=args.num_lags,
#                 num_features=num_features,
#                 exogenous_data=tmp_exogenous_data_train,
#                 indices=idxs,
#                 batch_size=1,
#                 shuffle=False,
#             )
#         )
#         val_loaders.append(
#             to_torch_dataset(
#                 client_X_val[client],
#                 client_y_val[client],
#                 num_lags=args.num_lags,
#                 exogenous_data=tmp_exogenous_data_val,
#                 indices=idxs,
#                 batch_size=1,
#                 shuffle=False,
#             )
#         )

#     cids = [k for k in client_X_train.keys() if k != "all"]

#     # 预测并逆归一化（每个子客户端单独逆归一化）
#     y_preds_train, y_preds_val = dict(), dict()
#     y_train_inv, y_val_inv, y_pred_train_inv, y_pred_val_inv = dict(), dict(), dict(), dict()
#     for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders):
#         print(f"Prediction on {cid}")
#         train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
#             model, train_loader, None, device=device
#         )
#         val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
#             model, val_loader, None, device=device
#         )
#         y_preds_train[cid] = y_pred_train
#         y_preds_val[cid] = y_pred_val

#         area = cid.split("_Client_")[0]
#         y_scaler = original_y_scalers[area]
#         y_train, y_val = client_y_train[cid], client_y_val[cid]
#         y_pred_train, y_pred_val = y_pred_train, y_pred_val
#         y_train, y_val, y_pred_train, y_pred_val = inverse_transform(
#             y_train, y_val, y_pred_train, y_pred_val,
#             y_scaler, round_preds=apply_round, dims=round_dimensions
#         )
#         y_train_inv[cid] = y_train
#         y_val_inv[cid] = y_val
#         y_pred_train_inv[cid] = y_pred_train
#         y_pred_val_inv[cid] = y_pred_val

#     # 合并到大area
#     train_metrics = []
#     val_metrics = []
#     for area in ["building1", "building2", "building3"]:
#         cids_this_area = [cid for cid in cids if cid.split("_Client_")[0] == area]
#         y_train = np.concatenate([y_train_inv[cid] for cid in cids_this_area], axis=0)
#         y_val = np.concatenate([y_val_inv[cid] for cid in cids_this_area], axis=0)
#         y_pred_train = np.concatenate([y_pred_train_inv[cid] for cid in cids_this_area], axis=0)
#         y_pred_val = np.concatenate([y_pred_val_inv[cid] for cid in cids_this_area], axis=0)

#         print("After inverse_transform:")
#         print("y_train[:5]:", y_train[:5])
#         print("y_pred_train[:5]:", y_pred_train[:5])

#         # 计算指标
#         train_mse, train_rmse, train_mae, train_r2, train_nrmse, train_res_per_dim = accumulate_metric(
#             y_train, y_pred_train, True, return_all=True
#         )
#         val_mse, val_rmse, val_mae, val_r2, val_nrmse, val_res_per_dim = accumulate_metric(
#             y_val, y_pred_val, True, return_all=True
#         )
#         train_metrics.append([train_mse, train_rmse, train_mae, train_r2, train_nrmse])
#         val_metrics.append([val_mse, val_rmse, val_mae, val_r2, val_nrmse])

#         log(INFO, f"\nFinal Prediction on {area} (Inference Stage)")
#         log(INFO, f"[Train]: mse: {train_mse}, "
#                   f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
#         log(INFO, f"[Val]: mse: {val_mse}, "
#                   f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}, nrmse: {val_nrmse}\n\n")

#         if plot:
#             make_plot(
#                 y_train,
#                 y_pred_train,
#                 title="Train",
#                 feature_names=args.targets,
#                 client=area,
#             )
#             make_plot(
#                 y_val,
#                 y_pred_val,
#                 title="Val",
#                 feature_names=args.targets,
#                 client=area,
#             )

#     # 转为numpy方便计算
#     train_metrics = np.array(train_metrics)
#     val_metrics = np.array(val_metrics)

#     # 计算平均
#     avg_train_metrics = train_metrics.mean(axis=0)
#     avg_val_metrics = val_metrics.mean(axis=0)

#     log(INFO, "\n=== 全局平均指标 (三地区平均) ===")
#     log(INFO, f"[Train] mse: {avg_train_metrics[0]}, rmse: {avg_train_metrics[1]}, mae: {avg_train_metrics[2]}, r2: {avg_train_metrics[3]}, nrmse: {avg_train_metrics[4]}")
#     log(INFO, f"[Val]   mse: {avg_val_metrics[0]}, rmse: {avg_val_metrics[1]}, mae: {avg_val_metrics[2]}, r2: {avg_val_metrics[3]}, nrmse: {avg_val_metrics[4]}")




def inference(
    model,  # the global model
    client_X_train,  # train data per client
    client_y_train,
    client_X_val,  # val data per client
    client_y_val,
    exogenous_data_train,  # exogenous data per client
    exogenous_data_val,
    original_y_scalers,  # 使用原始的 y_scalers
    idxs=[0],
    apply_round=False,  # 关闭四舍五入
    round_dimensions=[0],  # the dimensions to apply rounding
    plot=True,  # plot predictions
):
    import pandas as pd
    import os

    save_dir = "./results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load per client data to torch
    train_loaders, val_loaders = [], []

    # get data per client
    for client in client_X_train:
        if client == "all":
            continue
        if exogenous_data_train is not None:
            tmp_exogenous_data_train = exogenous_data_train[client]
            tmp_exogenous_data_val = exogenous_data_val[client]
        else:
            tmp_exogenous_data_train = None
            tmp_exogenous_data_val = None

        num_features = len(client_X_train[client][0][0])

        train_loaders.append(
            to_torch_dataset(
                client_X_train[client],
                client_y_train[client],
                num_lags=args.num_lags,
                num_features=num_features,
                exogenous_data=tmp_exogenous_data_train,
                indices=idxs,
                batch_size=1,
                shuffle=False,
            )
        )
        val_loaders.append(
            to_torch_dataset(
                client_X_val[client],
                client_y_val[client],
                num_lags=args.num_lags,
                exogenous_data=tmp_exogenous_data_val,
                indices=idxs,
                batch_size=1,
                shuffle=False,
            )
        )

    cids = [k for k in client_X_train.keys() if k != "all"]

    # 预测并逆归一化（每个子客户端单独逆归一化）
    y_preds_train, y_preds_val = dict(), dict()
    y_train_inv, y_val_inv, y_pred_train_inv, y_pred_val_inv = dict(), dict(), dict(), dict()
    for cid, train_loader, val_loader in zip(cids, train_loaders, val_loaders):
        print(f"Prediction on {cid}")
        train_mse, train_rmse, train_mae, train_r2, train_nrmse, y_pred_train = test(
            model, train_loader, None, device=device
        )
        val_mse, val_rmse, val_mae, val_r2, val_nrmse, y_pred_val = test(
            model, val_loader, None, device=device
        )
        y_preds_train[cid] = y_pred_train
        y_preds_val[cid] = y_pred_val

        area = cid.split("_Client_")[0]
        y_scaler = original_y_scalers[area]
        y_train, y_val = client_y_train[cid], client_y_val[cid]
        y_pred_train, y_pred_val = y_pred_train, y_pred_val
        y_train, y_val, y_pred_train, y_pred_val = inverse_transform(
            y_train, y_val, y_pred_train, y_pred_val,
            y_scaler, round_preds=apply_round, dims=round_dimensions
        )
        y_train_inv[cid] = y_train
        y_val_inv[cid] = y_val
        y_pred_train_inv[cid] = y_pred_train
        y_pred_val_inv[cid] = y_pred_val

    # 合并到大area
    train_metrics = []
    val_metrics = []
    summary_rows = []



    #新增
    area_map = {"building1": "area1", "building2": "area2", "building3": "area3"}
    for area in ["building1", "building2", "building3"]:



        area_label = area_map[area]




        cids_this_area = [cid for cid in cids if cid.split("_Client_")[0] == area]
        y_train = np.concatenate([y_train_inv[cid] for cid in cids_this_area], axis=0)
        y_val = np.concatenate([y_val_inv[cid] for cid in cids_this_area], axis=0)
        y_pred_train = np.concatenate([y_pred_train_inv[cid] for cid in cids_this_area], axis=0)
        y_pred_val = np.concatenate([y_pred_val_inv[cid] for cid in cids_this_area], axis=0)

        print("After inverse_transform:")
        print("y_train[:5]:", y_train[:5])
        print("y_pred_train[:5]:", y_pred_train[:5])

        # 保存真实值和预测值到csv
        df_result_train = pd.DataFrame({
            "y_true": y_train.flatten(),
            "y_pred": y_pred_train.flatten()
        })
        df_result_train.to_csv(os.path.join(save_dir, f"{area_label}_train_pred.csv"), index=False)

        df_result_val = pd.DataFrame({
            "y_true": y_val.flatten(),
            "y_pred": y_pred_val.flatten()
        })
        df_result_val.to_csv(os.path.join(save_dir, f"{area_label}_val_pred.csv"), index=False)

        # 计算指标
        train_mse, train_rmse, train_mae, train_r2, train_nrmse, train_res_per_dim = accumulate_metric(
            y_train, y_pred_train, True, return_all=True
        )
        val_mse, val_rmse, val_mae, val_r2, val_nrmse, val_res_per_dim = accumulate_metric(
            y_val, y_pred_val, True, return_all=True
        )
        train_metrics.append([train_mse, train_rmse, train_mae, train_r2, train_nrmse])
        val_metrics.append([val_mse, val_rmse, val_mae, val_r2, val_nrmse])

        summary_rows.append({
            "area": area_label,
            "train_mse": train_mse,
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2,
            "train_nrmse": train_nrmse,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "val_nrmse": val_nrmse,
        })

        log(INFO, f"\nFinal Prediction on {area} (Inference Stage)")
        log(INFO, f"[Train]: mse: {train_mse}, "
                  f"rmse: {train_rmse}, mae {train_mae}, r2: {train_r2}, nrmse: {train_nrmse}")
        log(INFO, f"[Val]: mse: {val_mse}, "
                  f"rmse: {val_rmse}, mae {val_mae}, r2: {val_r2}, nrmse: {val_nrmse}\n\n")

        if plot:
            make_plot(
                y_train,
                y_pred_train,
                title="Train",
                feature_names=args.targets,
                client=area_label,
            )
            make_plot(
                y_val,
                y_pred_val,
                title="Val",
                feature_names=args.targets,
                client=area_label,
            )

    # 转为numpy方便计算
    train_metrics = np.array(train_metrics)
    val_metrics = np.array(val_metrics)

    # 计算平均
    avg_train_metrics = train_metrics.mean(axis=0)
    avg_val_metrics = val_metrics.mean(axis=0)

    log(INFO, "\n=== 全局平均指标 (三地区平均) ===")
    log(INFO, f"[Train] mse: {avg_train_metrics[0]}, rmse: {avg_train_metrics[1]}, mae: {avg_train_metrics[2]}, r2: {avg_train_metrics[3]}, nrmse: {avg_train_metrics[4]}")
    log(INFO, f"[Val]   mse: {avg_val_metrics[0]}, rmse: {avg_val_metrics[1]}, mae: {avg_val_metrics[2]}, r2: {avg_val_metrics[3]}, nrmse: {avg_val_metrics[4]}")

    # 保存全局平均指标
    summary_rows.append({
        "area": "average",
        "train_mse": avg_train_metrics[0],
        "train_rmse": avg_train_metrics[1],
        "train_mae": avg_train_metrics[2],
        "train_r2": avg_train_metrics[3],
        "train_nrmse": avg_train_metrics[4],
        "val_mse": avg_val_metrics[0],
        "val_rmse": avg_val_metrics[1],
        "val_mae": avg_val_metrics[2],
        "val_r2": avg_val_metrics[3],
        "val_nrmse": avg_val_metrics[4],
    })
    pd.DataFrame(summary_rows).to_csv(os.path.join(save_dir, "强化学习.csv"), index=False)

inference(
    global_model,
    client_X_train, 
    client_y_train,
    client_X_val, 
    client_y_val,
    exogenous_data_train, 
    exogenous_data_val,
    original_y_scalers
)



