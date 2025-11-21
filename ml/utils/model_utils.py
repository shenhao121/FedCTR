from ml.models.rnn import RNN
from ml.models.lstm import LSTM
from ml.models.gru import GRU
from ml.models.cnn import CNN
from ml.models.rnn_autoencoder import DualAttentionAutoEncoder

from argparse import Namespace



args = Namespace(

    #新增
    data_path='../dataset/modified_data.csv', # dataset

    # data_path='../dataset/full_dataset.csv', # dataset

    test_size=0.2, # validation size 


    #新增
    target='meter_reading', # the target column to predict
    # targets=['rnti_count', 'rb_down', 'rb_up', 'down', 'up'], # the target columns





    num_lags=24, # the number of past observations to feed as input


    #新增
    identifier='building_id', # the column name that identifies a building
    # identifier='District', # the column name that identifies a bs

    nan_constant=0, # the constant to transform nan values
    x_scaler='minmax', # x_scaler
    y_scaler='minmax', # y_scaler

    #新增
    targets=["meter_reading"],
    
    #新增
    outlier_detection=False, # whether to perform flooring and capping
    # outlier_detection=True, # whether to perform flooring and capping

    criterion='mse', # optimization criterion, mse or l1
    fl_rounds=80, # the number of federated rounds
    fraction=1., # the percentage of available client to consider for random selection
    aggregation="ddpg", # federated aggregation algorithm
    epochs=10, # the number of maximum local epochs
    lr=0.001, # learning rate
    optimizer='adam', # the optimizer, it can be sgd or adam
    batch_size=128, # the batch size to use
    local_early_stopping=False, # whether to use early stopping
    local_patience=50, # patience value for the early stopping parameter (if specified)
    max_grad_norm=0.0, # whether to clip grad norm
    reg1=0.0, # l1 regularization
    reg2=0.0, # l2 regularization
    model_name='lstm', # the model to use, it can be rnn, lstm, gru, cnn or da_encoder_decoder
    cuda=True, # whether to use gpu
    
    seed=42, # reproducibility

    # here we define the exogenous data
    assign_stats=["mean", "median", "std", "variance", "kurtosis", "skew"],
    use_time_features=True, # whether to use datetime features

    # 新增对比学习开关
    use_contrastive=False  # 是否启用对比学习


)


# 定义 get_model 函数
def get_model(model: str,
              input_dim: int,
              out_dim: int,
              lags: int = 10,
              exogenous_dim: int = 0,
              seed=0):
    if model == "rnn":
        model = RNN(input_dim=input_dim, rnn_hidden_size=128, num_rnn_layers=1, rnn_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "lstm":
        model = LSTM(input_dim=input_dim, lstm_hidden_size=128, num_lstm_layers=1, lstm_dropout=0.0,
                     layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "gru":
        model = GRU(input_dim=input_dim, gru_hidden_size=128, num_gru_layers=1, gru_dropout=0.0,
                    layer_units=[128], num_outputs=out_dim, matrix_rep=True, exogenous_dim=exogenous_dim)
    elif model == "cnn":
        model = CNN(num_features=input_dim, lags=lags, exogenous_dim=exogenous_dim, out_dim=out_dim)
    elif model == "da_encoder_decoder":
        model = DualAttentionAutoEncoder(input_dim=input_dim, architecture="lstm", matrix_rep=True)
    else:
        raise NotImplementedError("Specified model is not implemented. Choose one from ['rnn', 'lstm', 'gru', 'cnn', 'da_encoder_decoder']")
    return model


# def get_input_dims(X_train, exogenous_data_train):
#     if args.model_name == "mlp":
#         input_dim = X_train.shape[1] * X_train.shape[2]
#     else:
#         input_dim = X_train.shape[2]
#
#     if exogenous_data_train is not None:
#         if len(exogenous_data_train) == 1:
#             cid = next(iter(exogenous_data_train.keys()))
#             exogenous_dim = exogenous_data_train[cid].shape[1]
#         else:
#             exogenous_dim = exogenous_data_train["all"].shape[1]
#     else:
#         exogenous_dim = 0
#
#     return input_dim, exogenous_dim
