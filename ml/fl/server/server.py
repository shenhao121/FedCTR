"""
Implements the server and the federated process.
"""
import scipy.stats
import copy
import sys

from pathlib import Path

import torch

from ml.fl.server.aggregation.DDPG import DDPG
from ml.fl.server.aggregation.aggregate import aggregate_with_ddpg

parent = Path(__file__).resolve().parents[3]
if parent not in sys.path:
    sys.path.insert(0, str(parent))

import time
from logging import DEBUG, INFO
from typing import Optional, Callable, List, Tuple, Dict, Union

import numpy as np
from torch.utils.data import DataLoader

from ml.fl.server.client_proxy import ClientProxy
from ml.fl.server.client_manager import ClientManager, SimpleClientManager

from ml.utils.logger import log
from ml.fl.history.history import History

from ml.fl.server.aggregation.aggregator import Aggregator
from ml.fl.defaults import weighted_loss_avg, weighted_metrics_avg




from ml.utils.model_utils import *

#暂时吧CNN放到这
class CNN(torch.nn.Module):
    def __init__(self,
                 num_features=11, lags=10, out_dim=1,
                 exogenous_dim: int = 0,
                 in_channels=[1, 16],
                 out_channels=[16, 32],
                 kernel_sizes=[(2, 3), (5, 3)],
                 pool_kernel_sizes=[(2, 1)]):
        super(CNN, self).__init__()
        assert len(in_channels) == len(out_channels) == len(kernel_sizes)
        self.activation = torch.nn.Tanh()
        self.num_lags = lags
        self.num_features = num_features
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0],
                                     kernel_size=kernel_sizes[0], padding="same")
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1],
                                     kernel_size=kernel_sizes[1], padding="same")
        self.pool = torch.nn.AvgPool2d(kernel_size=pool_kernel_sizes[0])
        kernel0, kernel1 = pool_kernel_sizes[-1][0], pool_kernel_sizes[-1][1]
        self.fc = torch.nn.Linear(
            in_features=(out_channels[1] * int(lags / kernel0) * int(num_features / kernel1)) + exogenous_dim,
            out_features=out_dim)

    def forward(self, x, exogenous_data=None, device=None, y_hist=None):
        if len(x.shape) > 2:
            x = x.view(x.size(0), x.size(3), x.size(1), x.size(2))
        else:
            x = x.view(x.size(0), 1, self.num_lags, self.num_features,)
        x = self.conv1(x)  # [batch_size]
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        # concatenate conv output with exogenous data
        if exogenous_data is not None and len(exogenous_data) > 0:
            x = torch.cat((x, exogenous_data), dim=1)

        x = self.fc(x)

        return x










class Server:
    def __init__(self,
                 client_proxies: List[ClientProxy],
                 X_train,  # 新增参数
                 exogenous_data_train,  # 新增参数
                 client_manager: Optional[ClientManager] = None,
                 aggregation: Optional[str] = None,
                 aggregation_params: Optional[Dict[str, Union[str, int, float, bool]]] = None,
                 weighted_loss_fn: Optional[Callable] = None,
                 weighted_metrics_fn: Optional[Callable] = None,
                 val_loader: Optional[DataLoader] = None,
                 local_params_fn: Optional[Callable] = None):


        self.X_train = X_train  # 保存传入的 X_train
        self.exogenous_data_train = exogenous_data_train  # 保存传入的 exogenous_data_train

        self.previous_weights = None  # 初始化 previous_weights

        # self.global_model = None
        self.global_model = self._initialize_global_model()
        self.best_model = None
        self.best_loss, self.best_epoch = np.inf, -1

        self.aggregation = aggregation

        
        self.client_proxies = client_proxies
        self._initialize_client_manager(client_manager)  # initialize the client manager

        self.weighted_loss = weighted_loss_fn if weighted_loss_fn is not None else weighted_loss_avg
        self.weighted_metrics = weighted_metrics_fn if weighted_metrics_fn is not None else weighted_metrics_avg

        if aggregation is None:
            aggregation = "fedavg"
        self.aggregator = Aggregator(aggregation_alg=aggregation, params=aggregation_params)
        log(INFO, f"Aggregation algorithm: {repr(self.aggregator)}")

        self.val_loader = val_loader
        self.local_params_fn = local_params_fn

        if aggregation == "ddpg":
            # 初始化 DDPG 实例
            self.ddpg_agent = DDPG(
                # state_dim=aggregation_params.get("state_dim", 10),  # 状态维度（如 mse、rmse 等）
                state_dim=10 if aggregation_params is None else aggregation_params.get("state_dim", 10),
                action_dim=len(client_proxies),  # 动作维度等于客户端数量
                device="cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 或 CPU
            )
            self.reward_history = []  # 用于存储奖励历史



    def _initialize_global_model(self) -> torch.nn.Module:
        """初始化全局模型"""
        model_name = args.model_name
        input_dim = 11
        out_dim = 1
        lags = args.num_lags
        exogenous_dim = 10
        seed = args.seed
        # input_dim, exogenous_dim = get_input_dims(self.X_train, self.exogenous_data_train)  # 动态获取 input_dim 和 exogenous_dim
        model = get_model(model=model_name,
                        input_dim=input_dim,
                        out_dim=out_dim,
                        lags=lags,
                        exogenous_dim=exogenous_dim,
                        seed=seed)

        return model

    def _initialize_client_manager(self, client_manager) -> None:
        """Initialize client manager""" 
        log(INFO, "Initializing client manager...")
        if client_manager is None:
            client_manager: ClientManager = SimpleClientManager()
            self.client_manager = client_manager
        else:
            self.client_manager = client_manager

        log(INFO, "Registering clients...")
        for client_proxy in self.client_proxies:  # register clients
            self.client_manager.register(client_proxy)

        log(INFO, "Client manager initialized!")

    def fit(self,
            num_rounds: int,
            fraction: float,
            fraction_args: Optional[Callable] = None,
            use_carbontracker: bool = False) -> Tuple[Dict[str, torch.Tensor], History]:
        """Run federated rounds for num_rounds rounds."""

        history = History()

        self.evaluate_round(fl_round=0, history=history)

        log(INFO, "Starting FL rounds")
        cb_tracker = None
        if use_carbontracker:
            try:
                from carbontracker.tracker import CarbonTracker
                cb_tracker = CarbonTracker(epochs=num_rounds, components="all", verbose=1)
            except ImportError:
                pass

        start_time = time.time()

        for fl_round in range(1, num_rounds + 1):
            if use_carbontracker and cb_tracker is not None:
                cb_tracker.epoch_start()
            # train and replace the previous global model
            self.fit_round(fl_round=fl_round,
                           fraction=fraction,
                           fraction_args=fraction_args,
                           history=history)
            if use_carbontracker and cb_tracker is not None:
                cb_tracker.epoch_end()
            # evaluate global model
            self.evaluate_round(fl_round=fl_round,
                                history=history)
        end_time = time.time()
        # log(INFO, history)
        log(INFO, f"Time passed: {end_time - start_time} seconds.")
        log(INFO, f"Best global model found on fl_round={self.best_epoch} with loss={self.best_loss}")

        # return self.best_model, history
        # return self.best_model.state_dict(), history
        return self.best_model, history

    def fit_round(self, fl_round: int,
                  fraction: float,
                  fraction_args: Optional[Callable],
                  history: History) -> None:
        """Perform a federated round, i.e.,
            1) Select a fraction of available clients.
            2) Instruct selected clients to execute local training.
            3) Receive updated parameters from clients and their corresponding evaluation
            4) Aggregate the local learned weights.
        """
        # Inform clients for local parameters change if any
        if self.local_params_fn:
            for client_proxy in self.client_proxies:
                client_proxy.set_train_parameters(self.local_params_fn(fl_round), verbose=True)

        # STEP 1: Select a fraction of available clients
        selected_clients = self.sample_clients(fl_round, fraction, fraction_args)

        # STEPS 2-3: Perform local training and receive updated parameters
        num_train_examples: List[int] = []
        num_test_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        test_losses: Dict[str, float] = dict()
        all_train_metrics: Dict[str, Dict[str, float]] = dict()
        all_test_metrics: Dict[str, Dict[str, float]] = dict()
        results: List[Tuple[List[np.ndarray], int]] = []

        for client in selected_clients:
            res = self.fit_client(fl_round, client)
            model_params, num_train, train_loss, train_metrics, num_test, test_loss, test_metrics = res
            num_train_examples.append(num_train)
            num_test_examples.append(num_test)
            train_losses[client.cid] = train_loss
            test_losses[client.cid] = test_loss
            all_train_metrics[client.cid] = train_metrics
            all_test_metrics[client.cid] = test_metrics
            results.append((model_params, num_train))

        history.add_local_train_loss(train_losses, fl_round)
        history.add_local_train_metrics(all_train_metrics, fl_round)
        history.add_local_test_loss(test_losses, fl_round)
        history.add_local_test_metrics(all_test_metrics, fl_round)

        # # STEP 4: Aggregate local models
        # self.global_model = self.aggregate_models(fl_round, results)

        if self.aggregation == "ddpg":
            #新增强化学习聚合
            # STEP 4: Aggregate local models
            self.global_model = self.aggregate_models2(fl_round, results, history)
        else:
            self.global_model = self.aggregate_models(fl_round, results)



        if self.best_model is None:
            self.best_model = copy.deepcopy(self.global_model)

    def sample_clients(self, fl_round: int, fraction: float,
                       fraction_args: Optional[Callable] = None) -> List[ClientProxy]:
        """Sample available clients."""
        if fraction_args is not None:
            fraction: float = fraction_args(fl_round)

        selected_clients: List[ClientProxy] = self.client_manager.sample(fraction)
        #log(DEBUG, f"[Global round {fl_round}] Sampled {len(selected_clients)} clients "
        #           f"(out of {self.client_manager.num_available(verbose=False)})")

        return selected_clients

    def fit_client(self,
                   fl_round: int,
                   client: ClientProxy) -> Tuple[
        List[np.ndarray], int, float, Dict[str, float], int, float, Dict[str, float]]:
        """Perform local training."""
        #log(INFO, f"[Global round {fl_round}] Fitting client {client.cid}")
        if fl_round == 1:
            fit_res = client.fit(None)
        else:
            fit_res = client.fit(model=self.global_model)

        return fit_res

    def aggregate_models(self, fl_round: int, results: List[Tuple[List[np.ndarray], int]]) -> torch.nn.Module:
        log(INFO, f"[Global round {fl_round}] Aggregating local models...")
        # aggregated_params = self.aggregator.aggregate(results, self.global_model)
        self.global_model = self.aggregator.aggregate(results, self.global_model)

        return self.global_model





    def aggregate_models2(self, fl_round: int, results: List[Tuple[List[np.ndarray], int]], history: History) -> torch.nn.Module:
        log(INFO, f"[Global round {fl_round}] Aggregating local models using DDPG...")

        # 收集客户端状态特征
        client_features = []
        for client_id, (weights, num_samples) in enumerate(results):
            client_model_weights = self.client_proxies[client_id].get_parameters()
            _, loss, metrics = self.client_proxies[client_id].evaluate(method="test")
            mse = metrics["MSE"]
            rmse = metrics["RMSE"]
            mae = metrics["MAE"]
            r2 = metrics["R^2"]
            nrmse = metrics["NRMSE"]
            data_ratio = num_samples / sum([r[1] for r in results])
            # 定义 client_data 为客户端模型参数的展平数组
            client_data = np.concatenate([param.flatten() for param in client_model_weights])            #新增
            mean = np.mean(client_data)
            std = np.std(client_data)
            skewness = scipy.stats.skew(client_data)
            kurtosis = scipy.stats.kurtosis(client_data)

            client_features.append([mse, rmse, mae, r2, nrmse, data_ratio,mean, std, skewness, kurtosis])
            

        # client_models = [self.client_proxies[client_id].get_parameters() for client_id in range(len(self.client_proxies))]
        client_models = []
        for client_id in range(len(self.client_proxies)):
            model = copy.deepcopy(self.global_model)  # 创建一个新的模型对象
            client_model_params = self.client_proxies[client_id].get_parameters()  # 获取客户端参数
            model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), client_model_params)})  # 加载参数
            client_models.append(model)  # 将模型添加到列表中
        
        global_model = self.global_model
        new_global_w, client_weights = aggregate_with_ddpg(
            self.ddpg_agent,
            client_features,
            client_models,
            global_model,
            previous_weights=self.previous_weights  # 传递 previous_weights
        )
        self.previous_weights = client_weights  # 更新 previous_weights
        log(INFO, f"Client weights for this round: {client_weights}")


        # === 新增：下发 RL 权重给每个客户端 ===
        base_contrastive_lambda = 0.001  # 你可以根据需要调整
        for cid, client_proxy in enumerate(self.client_proxies):
            # 只有ddpg聚合时才下发动态权重
            if args.aggregation == "ddpg":
                contrastive_lambda = base_contrastive_lambda * client_weights[cid]
            else:
                contrastive_lambda = base_contrastive_lambda
            client_proxy.set_train_parameters({'contrastive_lambda': float(contrastive_lambda)})






        # 更新 global_model
        global_model = copy.deepcopy(self.global_model)
        global_model.load_state_dict(new_global_w)


        # 定义 reward
        previous_global_loss = history.global_test_losses[-1] if len(history.global_test_losses) > 0 else np.inf
        previous_val_r2 = history.global_test_metrics["R^2"][-1] if len(history.global_test_metrics["R^2"]) > 0 else -np.inf

        self.global_model = global_model  # 更新全局模型
        self.evaluate_round(fl_round, history)  # 使用 evaluate_round 方法评估全局模型
        current_global_loss = history.global_test_losses[-1]  # 从 history 中获取最新的测试损失
        current_val_r2 = history.global_test_metrics["R^2"][-1]

        # reward = previous_global_loss - current_global_loss
        reward = (previous_global_loss - current_global_loss) + 0.5 * (current_val_r2 - previous_val_r2)


        # 定义 action 和 next_state
        action = client_weights  # 动作是客户端权重



        # next_state = client_features
        # next_state = self.calculate_next_state(global_model, client_features)
        next_state = self.calculate_next_state(global_model, results)  # 基于聚合后的全局模型测试结果生成下一状态


        client_features = np.array(client_features).mean(axis=0)  # 将维度从 [num_clients, state_dim] 转换为 [state_dim]


        # 存储经验
        self.ddpg_agent.save_experience(client_features, action, reward, next_state)

        # 更新强化学习模型
        self.ddpg_agent.update()


        return global_model


    def calculate_next_state(self, global_model, results):
        """基于聚合后的全局模型测试结果生成下一状态"""
        next_state = []
        for client_id, (weights, num_samples) in enumerate(results):
            _, loss, metrics = self.client_proxies[client_id].evaluate(model=global_model, method="test")
            mse = metrics["MSE"]
            rmse = metrics["RMSE"]
            mae = metrics["MAE"]
            r2 = metrics["R^2"]
            nrmse = metrics["NRMSE"]
            data_ratio = num_samples / sum([r[1] for r in results])

              # 获取客户端模型参数并计算新增指标
            client_model_weights = self.client_proxies[client_id].get_parameters()
            client_data = np.concatenate([param.flatten() for param in client_model_weights])
            mean = np.mean(client_data)
            std = np.std(client_data)
            skewness = scipy.stats.skew(client_data)
            kurtosis = scipy.stats.kurtosis(client_data)


            next_state.append([mse, rmse, mae, r2, nrmse, data_ratio, mean, std, skewness, kurtosis])  # 将测试结果作为下一状态
        return np.array(next_state).mean(axis=0)  # 将维度从 [num_clients, state_dim] 转换为 [state_dim]


    def evaluate_round(self, fl_round: int, history: History):
        """Evaluate global model."""
        num_train_examples: List[int] = []
        train_losses: Dict[str, float] = dict()
        train_metrics: Dict[str, Dict[str, float]] = dict()
        num_test_examples: List[int] = []
        test_losses: Dict[str, float] = dict()
        test_metrics: Dict[str, Dict[str, float]] = dict()

        if fl_round == 0:
            #log(INFO, "Evaluating initial global model")
            self.global_model: List[np.ndarray] = self._get_initial_model()

        if self.val_loader:
            random_client = self.client_manager.sample(0.)[0]
            num_instances, loss, eval_metrics = random_client.evaluate(data=self.val_loader, model=self.global_model)
            num_test_examples = [num_instances]
            test_metrics["Server"] = eval_metrics
            test_losses["Server"] = loss

        else:
            for cid, client_proxy in self.client_manager.all().items():
                num_train_instances, train_loss, train_eval_metrics = client_proxy.evaluate(model=self.global_model,
                                                                                            method="train")

                num_train_examples.append(num_train_instances)
                train_losses[cid] = train_loss
                train_metrics[cid] = train_eval_metrics

                num_test_instances, test_loss, test_eval_metrics = client_proxy.evaluate(model=self.global_model,
                                                                                         method="test")
                num_test_examples.append(num_test_instances)
                test_losses[cid] = test_loss
                test_metrics[cid] = test_eval_metrics

        history.add_global_train_losses(self.weighted_loss(num_train_examples, list(train_losses.values())))
        history.add_global_train_metrics(self.weighted_metrics(num_train_examples, train_metrics))

        history.add_global_test_losses(self.weighted_loss(num_test_examples, list(test_losses.values())))


          # 提取验证集的 r2 值
        if "Server" in test_metrics and "R^2" in test_metrics["Server"]:
            current_val_r2 = test_metrics["Server"]["R^2"]
            history.global_test_metrics["R^2"].append(current_val_r2)




        if history.global_test_losses[-1] <= self.best_loss:
            #log(DEBUG, f"Caching best global model, fl_round={fl_round}")
            self.best_loss = history.global_test_losses[-1]
            self.best_epoch = fl_round
            self.best_model = copy.deepcopy(self.global_model)

        history.add_global_test_metrics(self.weighted_metrics(num_test_examples, test_metrics))

    # def _get_initial_model(self) -> List[np.ndarray]:
    #     """Get initial parameters from a random client"""
    #     random_client = self.client_manager.sample(0.)[0]
    #     client_model = random_client.get_parameters()
    #     # log(INFO, "Received initial parameters from one random client!")
    #     return client_model
    def _get_initial_model(self) -> torch.nn.Module:
        """Get initial parameters from a random client"""
        random_client = self.client_manager.sample(0.)[0]
        client_model_params = random_client.get_parameters()
          # 确保 self.global_model 已正确初始化
        if self.global_model is None:
            raise ValueError("self.global_model 未正确初始化，请检查 Server 类的初始化逻辑。")

        model = copy.deepcopy(self.global_model)  # 假设 self.global_model 是一个 torch.nn.Module 对象
        model.load_state_dict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), client_model_params)})
        return model