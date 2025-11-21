"""
Aggregation functions.
"""

import copy
import math
from typing import List, Tuple
from functools import reduce






import torch
import numpy as np

# 新增模型聚合函数
def aggregate_with_ddpg(agent, client_features, client_models, global_model, previous_weights=None):
    """
    使用 DDPG 进行聚合。
    :param agent: DDPG 实例
    :param client_features: 各客户端状态特征列表
    :param client_models: 参与本轮聚合的客户端模型列表
    :param global_model: 全局模型
    :return: 新的全局模型参数和客户端权重
    """

    if not isinstance(global_model, torch.nn.Module):
        raise TypeError("global_model 应该是一个 torch.nn.Module 对象，而不是 list 类型。")

    #  # 特征归一化（如果需要）
    # states = np.array(client_features)
    # weights = agent.get_action(states.mean(axis=0))  # 使用平均状态特征
    
    states = np.array(client_features).mean(axis=0)  # 将维度从 [num_clients, state_dim] 转换为 [state_dim]
   

    weights = agent.get_action(states)  # 使用平均状态特征
    # 权重归一化
    weights = np.clip(weights, 0.01, None)  # 防止权重为0
    if previous_weights is not None:
        weights = 0.9 * weights + 0.1 * previous_weights  # 平滑处理
    weights /= weights.sum()

    # 执行模型聚合
    global_w = global_model.state_dict()
    for key in global_w.keys():
        global_w[key] = torch.zeros_like(global_w[key])

    for idx, (net, weight) in enumerate(zip(client_models, weights)):
        net_para = net.state_dict()
        for key in global_w:
            global_w[key] += net_para[key] * weight

    return global_w, weights


# def simple_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> np.ndarray:
def simple_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module) -> torch.nn.Module:
    """Compute a simple average."""
    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

     # 计算每一层的简单平均值
    weights_prime: List[np.ndarray] = [
        reduce(np.add, layer_updates) / len(weights)
        for layer_updates in zip(*weights)
    ]
       # 将 weights_prime 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model


# def median_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> np.ndarray:
#     """Compute median across weights."""
#     weights = [
#         [layer for layer in weights] for weights, _ in results
#     ]

#     weights_prime: np.ndarray = [
#         np.median(layer_updates, axis=0)
#         for layer_updates in zip(*weights)
#     ]
#     return weights_prime
def median_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module) -> torch.nn.Module:
    """Compute median across weights and return a torch.nn.Module."""
    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    weights_prime: List[np.ndarray] = [
        np.median(layer_updates, axis=0)
        for layer_updates in zip(*weights)
    ]

    # 将 weights_prime 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model

# def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> np.ndarray:
# def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]]) -> torch.nn.Module:

#     """Compute weighted average."""
#     # Calculate the total number of examples used during training
#     num_examples_total = sum([num_examples for _, num_examples in results])

#     # Create a list of weights, each multiplied by the related number of examples
#     weighted_weights = [
#         [layer * num_examples for layer in weights] for weights, num_examples in results
#     ]

#     # Compute average weights of each layer
#     weights_prime: np.ndarray = [
#         reduce(np.add, layer_updates) / num_examples_total
#         for layer_updates in zip(*weighted_weights)
#     ]
#     return weights_prime
def fedavg_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module) -> torch.nn.Module:
    """Compute weighted average and return a torch.nn.Module."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: List[np.ndarray] = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    # 将 weights_prime 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model

# def fednova_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
#                       rho: float = 0.) -> List[np.ndarray]:
#     """Compute weighted average according to FedNova."""
#     num_examples = [num_examples for _, num_examples in results]
#     num_examples_total = sum(num_examples)

#     weights = [
#         [layer for layer in weights] for weights, _ in results
#     ]

#     taus = copy.deepcopy(num_examples)
#     alphas = [taus[i] - rho * (1 - math.pow(rho, taus[i])) / (1 - rho) / (1 - rho) for i in range(len(taus))]

#     diffs = copy.deepcopy(weights)
#     for i in range(len(weights)):
#         for j in range(len(weights[i])):
#             diffs[i][j] = (previous_model[j] - weights[i][j]) / alphas[i]

#     d_total_round = [np.zeros_like(previous_model[i]) for i in range(len(previous_model))]

#     for i in range(len(diffs)):
#         d_para = diffs[i]
#         for j in range(len(diffs[i])):
#             d_total_round[j] = np.add(d_total_round[j], d_para[j] * num_examples[i] / num_examples_total)

#     coeff = 0.
#     for i in range(len(diffs)):
#         coeff = np.add(coeff, alphas[i] * num_examples[i] / num_examples_total)

#     weights_prime: List[np.ndarray] = copy.deepcopy(previous_model)
#     for i in range(len(weights_prime)):
#         weights_prime[i] = np.subtract(weights_prime[i], coeff * d_total_round[i])

#     return weights_prime
def fednova_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module, rho: float = 0.) -> torch.nn.Module:
    """Compute weighted average according to FedNova and return a torch.nn.Module."""
    # 原有逻辑保持不变
    num_examples = [num_examples for _, num_examples in results]
    num_examples_total = sum(num_examples)

    weights = [
        [layer for layer in weights] for weights, _ in results
    ]

    taus = copy.deepcopy(num_examples)
    alphas = [taus[i] - rho * (1 - math.pow(rho, taus[i])) / (1 - rho) / (1 - rho) for i in range(len(taus))]

    diffs = copy.deepcopy(weights)
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            diffs[i][j] = (global_model.state_dict()[list(global_model.state_dict().keys())[j]] - weights[i][j]) / alphas[i]

    d_total_round = [np.zeros_like(global_model.state_dict()[key]) for key in global_model.state_dict().keys()]

    for i in range(len(diffs)):
        d_para = diffs[i]
        for j in range(len(diffs[i])):
            d_total_round[j] = np.add(d_total_round[j], d_para[j] * num_examples[i] / num_examples_total)

    coeff = 0.
    for i in range(len(diffs)):
        coeff = np.add(coeff, alphas[i] * num_examples[i] / num_examples_total)

    weights_prime: List[np.ndarray] = copy.deepcopy(list(global_model.state_dict().values()))
    for i in range(len(weights_prime)):
        weights_prime[i] = np.subtract(weights_prime[i], coeff * d_total_round[i])

    # 将 weights_prime 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), weights_prime)})
    return new_global_model

# def fedadagrad_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
#                          m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
#                          beta_1: float = 0., eta: float = 0.1,
#                          tau: float = 1e-2) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
#     """Computed weighted average according to FedAdagrad."""
#     fedavg_aggregated = fedavg_aggregate(results)

#     delta_t: List[np.ndarray] = [
#         x - y for x, y in zip(fedavg_aggregated, previous_model)
#     ]

#     if not m_t:
#         m_t = [np.zeros_like(x) for x in delta_t]
#     m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

#     if not v_t:
#         v_t = [np.zeros_like(x) for x in delta_t]
#     v_t = [x + np.multiply(y, y) for x, y in zip(v_t, delta_t)]

#     new_weights = [
#         x + eta * y / (np.sqrt(z) + tau)
#         for x, y, z in zip(previous_model, m_t, v_t)
#     ]
#     return new_weights, m_t, v_t
def fedadagrad_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                         m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                         beta_1: float = 0., eta: float = 0.1, tau: float = 1e-2) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray]]:
    """Computed weighted average according to FedAdagrad and return a torch.nn.Module."""
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        x - y for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [x + np.multiply(y, y) for x, y in zip(v_t, delta_t)]

    new_weights = [
        x + eta * y / (np.sqrt(z) + tau)
        for x, y, z in zip(list(global_model.state_dict().values()), m_t, v_t)
    ]

    # 将 new_weights 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, m_t, v_t

# def fedyogi_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
#                       m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
#                       beta_1: float = 0.9, beta_2: float = 0.99,
#                       eta: float = 0.01,
#                       tau: float = 1e-3) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
#     """Compute weighted average according to FedYogi."""
#     fedavg_aggregated = fedavg_aggregate(results)

#     delta_t: List[np.ndarray] = [
#         x - y for x, y in zip(fedavg_aggregated, previous_model)
#     ]

#     if not m_t:
#         m_t = [np.zeros_like(x) for x in delta_t]
#     m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

#     if not v_t:
#         v_t = [np.zeros_like(x) for x in delta_t]
#     v_t = [x - (1.0 - beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
#            for x, y in zip(v_t, delta_t)]

#     new_weights = [
#         x + eta * y / (np.sqrt(z) + tau)
#         for x, y, z in zip(previous_model, m_t, v_t)
#     ]
#     return new_weights, m_t, v_t
def fedyogi_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                      m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                      beta_1: float = 0.9, beta_2: float = 0.99, eta: float = 0.01, tau: float = 1e-3) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray]]:
    """Compute weighted average according to FedYogi and return a torch.nn.Module."""
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        np.array(x) - np.array(y) for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    # 强制类型转换，确保 x 和 y 都是 numpy 数组
    m_t = [np.multiply(beta_1, np.array(x)) + (1 - beta_1) * np.array(y) for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [np.array(x) - (1 - beta_2) * np.multiply(np.array(y), np.array(y)) for x, y in zip(v_t, delta_t)]

    new_weights = [
        np.array(x) + eta * np.array(y) / (np.sqrt(np.array(z) + tau))
        for x, y, z in zip(list(global_model.state_dict().values()), m_t, v_t)
    ]

    # 将 new_weights 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, m_t, v_t

# def fedadam_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
#                       m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
#                       beta_1: float = 0.9, beta_2: float = 0.99,
#                       eta: float = 0.01,
#                       tau: float = 1e-3) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
#     """Compute weighted average according to FedAdam."""
#     fedavg_aggregated = fedavg_aggregate(results)

#     delta_t: List[np.ndarray] = [
#         x - y for x, y in zip(fedavg_aggregated, previous_model)
#     ]

#     if not m_t:
#         m_t = [np.zeros_like(x) for x in delta_t]
#     m_t = [np.multiply(beta_1, x) + (1 - beta_1) * y for x, y in zip(m_t, delta_t)]

#     if not v_t:
#         v_t = [np.zeros_like(x) for x in delta_t]
#     v_t = [beta_2 * x + (1 - beta_2) * np.multiply(y, y)
#            for x, y in zip(v_t, delta_t)]

#     new_weights = [
#         x + eta * y / (np.sqrt(z) + tau)
#         for x, y, z in zip(previous_model, m_t, v_t)
#     ]
#     return new_weights, m_t, v_t
def fedadam_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                      m_t: List[np.ndarray] = None, v_t: List[np.ndarray] = None,
                      beta_1: float = 0.9, beta_2: float = 0.99, eta: float = 0.01, tau: float = 1e-3) -> Tuple[torch.nn.Module, List[np.ndarray], List[np.ndarray]]:
    """Compute weighted average according to FedAdam and return a torch.nn.Module."""
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        np.array(x) - np.array(y) for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if not m_t:
        m_t = [np.zeros_like(x) for x in delta_t]
    # 强制类型转换，确保 x 和 y 都是 numpy 数组
    m_t = [np.multiply(beta_1, np.array(x)) + (1 - beta_1) * np.array(y) for x, y in zip(m_t, delta_t)]

    if not v_t:
        v_t = [np.zeros_like(x) for x in delta_t]
    v_t = [np.multiply(beta_2, np.array(x)) + (1 - beta_2) * np.multiply(np.array(y), np.array(y)) for x, y in zip(v_t, delta_t)]

    new_weights = [
        np.array(x) + eta * np.array(y) / (np.sqrt(np.array(z)) + tau)
        for x, y, z in zip(list(global_model.state_dict().values()), m_t, v_t)
    ]

    # 将 new_weights 转换为 torch.nn.Module
    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, m_t, v_t



# def fedavgm_aggregate(results: List[Tuple[List[np.ndarray], int]], previous_model: List[np.ndarray],
#                       server_momentum: float = 0., momentum_vector: List[np.ndarray] = None,
#                       server_lr: float = 1.) -> Tuple[List[np.ndarray], List[np.ndarray]]:
#     """Compute weighted average according to FedAvgM."""
#     fedavg_aggregated = fedavg_aggregate(results)

#     pseudo_gradient: List[np.ndarray] = [
#         x - y for x, y in zip(previous_model, fedavg_aggregated)
#     ]
#     if server_momentum > 0.0:
#         if momentum_vector is not None:
#             momentum_vector = [
#                 server_momentum * x + y
#                 for x, y in zip(momentum_vector, pseudo_gradient)
#             ]
#         else:
#             momentum_vector = pseudo_gradient
#         pseudo_gradient = copy.deepcopy(momentum_vector)

#     new_weights = [
#         x - server_lr * y
#         for x, y in zip(previous_model, pseudo_gradient)
#     ]

#     return new_weights, pseudo_gradient
# def fedavgm_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
#                       server_momentum: float = 0.9, momentum_vector: List[np.ndarray] = None, server_lr: float = 1.) -> Tuple[torch.nn.Module, List[np.ndarray]]:
#     """Compute weighted average according to FedAvgM and return a torch.nn.Module."""
#     fedavg_aggregated = fedavg_aggregate(results, global_model)

#     delta_t: List[np.ndarray] = [
#         x - y for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
#     ]

#     if not momentum_vector:
#         momentum_vector = [np.zeros_like(x) for x in delta_t]
#     momentum_vector = [server_momentum * x + y for x, y in zip(momentum_vector, delta_t)]

#     new_weights = [
#         x + server_lr * y
#         for x, y in zip(list(global_model.state_dict().values()), momentum_vector)
#     ]

#     # 将 new_weights 转换为 torch.nn.Module
#     new_global_model = copy.deepcopy(global_model)
#     new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
#     return new_global_model, momentum_vector


def fedavgm_aggregate(results: List[Tuple[List[np.ndarray], int]], global_model: torch.nn.Module,
                      server_momentum: float = 0.9, momentum_vector: List[np.ndarray] = None, server_lr: float = 1.) -> Tuple[torch.nn.Module, List[np.ndarray]]:
    fedavg_aggregated = fedavg_aggregate(results, global_model)

    delta_t: List[np.ndarray] = [
        np.array(x) - np.array(y) for x, y in zip(list(fedavg_aggregated.state_dict().values()), list(global_model.state_dict().values()))
    ]

    if momentum_vector is None:
        momentum_vector = [np.zeros_like(x) for x in delta_t]
    # 强制类型转换，确保 x 和 y 都是 numpy 数组
    momentum_vector = [server_momentum * np.array(x) + np.array(y) for x, y in zip(momentum_vector, delta_t)]

    new_weights = [
        np.array(x) - server_lr * np.array(y)
        for x, y in zip(list(global_model.state_dict().values()), momentum_vector)
    ]

    new_global_model = copy.deepcopy(global_model)
    new_global_model.load_state_dict({k: torch.tensor(v) for k, v in zip(global_model.state_dict().keys(), new_weights)})
    return new_global_model, momentum_vector