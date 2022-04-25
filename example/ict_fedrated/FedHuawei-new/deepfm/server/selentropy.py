import numpy as np
import random
import xlwt
from src.dataset import create_dataset, DataType
from client import Client
from mindspore import load_param_into_net, Parameter
from copy import deepcopy

class SelEntropy(object):

    def __init__(self, config):
        self.config = config

    # Set up server
    def boot(self):
        print('Booting {} server...'.format(self.config.fl_mode))
        config = self.config

        dataset_all = create_dataset(config.dataset_path,
                                     train_mode=True,
                                     epochs=1,
                                     batch_size=config.batch_size,
                                     data_type=DataType(config.data_format),
                                     rank_size=config.rank_size,
                                     rank_id=None, indexes=None)
        all_size = dataset_all.get_dataset_size()

        indexes = np.random.randint(0, all_size, (config.num_clients, config.num_samples))
        train_samples = round(config.num_samples * config.train_ratio)
        client_datasets = []
        for index in indexes:
            train_dataset = create_dataset(config.dataset_path,
                                           train_mode=True,
                                           epochs=1,
                                           batch_size=config.batch_size,
                                           data_type=DataType(config.data_format),
                                           rank_size=config.rank_size,
                                           rank_id=None, indexes=index[:train_samples])
            test_dataset = create_dataset(config.dataset_path,
                                          train_mode=False,
                                          epochs=1,
                                          batch_size=config.batch_size,
                                          data_type=DataType(config.data_format),
                                          indexes=index[train_samples:])
            # steps_size = dataset.get_dataset_size()
            client_datasets.append([train_dataset, test_dataset])
        # print(datasets)
        clients = []
        for idx in range(config.num_clients):
            client = Client(client_id=idx, config=config, train_set=client_datasets[idx][0],
                            test_set=client_datasets[idx][1])
            clients.append(client)
        self.clients = clients
        print('Total clients: {}'.format(config.num_clients))
    # Run federated learning
    def run(self):
        rounds = self.config.max_round
        workbook = xlwt.Workbook(encoding='ascii')
        aggsheet = workbook.add_sheet('agg')
        aggsheet.write(0, 0, label='轮')
        aggsheet.write(0, 1, label='精度')
        aggsheet.write(0, 2, label='loss')
        aggsheet.write(0, 3, label='通信量')
        aggsheet.write(0, 4, label='训练客户端数')
        aggsheet.write(0, 5, label='聚合客户端数')

        # Perform rounds of federated learning
        for round in range(1, rounds + 1):
            aggsheet.write(round, 0, round)

            print('**** Round {}/{} ****'.format(round, rounds))

            # Run the federated learning round
            train_log, com_capa, num_train_clients, num_agg_clients = self.train_round(round)
            print('Training finishes!')
            state_dict = deepcopy(train_log['state_dict'])
            train_accuracy = train_log['log']['res']
            print('Accuracy is : {}'.format(train_accuracy))
            aggsheet.write(round, 1, train_accuracy)
            aggsheet.write(round, 2, 0)
            aggsheet.write(round, 3, com_capa)
            aggsheet.write(round, 4, num_train_clients)
            aggsheet.write(round, 5, num_agg_clients)

            if round == 1:
                self.maxaccuracy = train_accuracy
                self.bestmodel = deepcopy(state_dict)
                self.bestround = 1
            elif self.maxaccuracy < train_accuracy:
                self.maxaccuracy = train_accuracy
                self.bestmodel = deepcopy(state_dict)
                self.bestround = round
        aggsheet.write(0, 6, label='最佳轮次')
        aggsheet.write(1, 6, self.bestround)

        workbook.save(self.config.fl_mode + '_' + str(self.config.num_clients) + '_' + str(
            self.config.num_client_per_round) + '_.xls')
        print('Saved excel: ' + self.config.fl_mode + '_' + str(self.config.num_clients) + '_' + str(
            self.config.num_client_per_round) + '_.xls')

    def train_round(self, round):
        # Select clients to participate in the round
        sample_have_clients, sample_not_clients, not_weight_clients, com_capa = self.selection(round)

        # 计算训练和聚合的客户端个数
        num_agg_clients = len(sample_have_clients) + len(sample_not_clients)
        num_train_clients = len(sample_have_clients) + len(not_weight_clients)

        local_results = []
        if round == 1:
            for client in sample_have_clients:
                client.local_execute(state_dict_to_load=None)
                client.set_round_time(round)
                local_results.append(client.local_result)
        else:
            for client in sample_have_clients:
                client.local_execute(state_dict_to_load=self.global_model)
                client.set_round_time(round)
                local_results.append(client.local_result)
        local_results += [client.local_result for client in sample_not_clients]

        agg_local_results = self.aggregate_local_results(local_results)
        com_capa += num_agg_clients * agg_local_results['num_params']
        self.global_model = deepcopy(agg_local_results['state_dict'])
        return agg_local_results, com_capa, num_train_clients, num_agg_clients

    def aggregate_local_results(self, local_results):
        num_parameters, state_dict = self.aggregate_local_state_dicts(
            [ltr['state_dict'] for ltr in local_results])
        return {
            'state_dict': state_dict,
            'num_params': num_parameters,
            'log': self.aggregate_local_logs(
                [ltr['log'] for ltr in local_results]
            )}

    def aggregate_local_state_dicts(self, local_train_state_dicts):
        num_parameters = 0
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:  # 模型有几部分
            agg_state_dict[k] = 0  # 每部分初始值为0
            for lstd in local_train_state_dicts:  # 获取每一个client的本地模型
                agg_state_dict[k] += lstd[k]  # 对本地模型的每一部分作和
            agg_state_dict[k] /= len(local_train_state_dicts)
            num_parameters += agg_state_dict[k].flatten().shape[0]
            agg_state_dict[k] = Parameter(default_input=agg_state_dict[k], name=k)
        return num_parameters, agg_state_dict

    def aggregate_local_logs(self, local_logs):
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log_idx, local_log in enumerate(local_logs):
                agg_log[k] += local_log[k] * self.config.num_samples
        for k in agg_log:
            agg_log[k] /= self.config.num_samples * len(local_logs)
        return agg_log

    def selection(self, round):
        # Select devices to participate in round
        clients_per_round = self.config.num_client_per_round
        if round == 1:
            # Select clients randomly
            sample_clients = [client for client in random.sample(
                self.clients, clients_per_round)]
            return sample_clients, [], [], 0
        else:
            # 先进行一步选择
            sample_clients = [client for client in random.sample(
                self.clients, clients_per_round)]
            have_weight_clients = []#有本地模型的参与者
            not_weight_clients = []#没有本地模型的参与者
            for client in sample_clients:
                if client.round_time != 0:
                    have_weight_clients.append(client)
                else:
                    not_weight_clients.append(client)
            # tar_clients = have_weight_clients + not_weight_clients
            for client in not_weight_clients:
                client.local_execute(state_dict_to_load=self.global_model)
                client.set_round_time(round)

            have_weight_clients = np.array(have_weight_clients)
            not_weight_clients = np.array(not_weight_clients)

            weights = [deepcopy(client.local_result['state_dict']) for client in have_weight_clients]
            weights = weights + [deepcopy(client.local_result['state_dict']) for client in not_weight_clients]

            bins = 100
            mulinfos = self.cal_mulinfo(weights, bins)
            com_capa = len(weights)
            probas = self.cal_proba(mulinfos)
            have_weight_probas = probas[0:len(have_weight_clients)]
            not_weight_probas = probas[len(have_weight_clients):]

            sample_have_clients = have_weight_clients[have_weight_probas > 0 ]#根据概率选择出客户端
            sample_not_clients = not_weight_clients[not_weight_probas > 0]
            return sample_have_clients, sample_not_clients, not_weight_clients, com_capa

    # calculate mutual information
    def cal_mulinfo(self, weights, bins):
        # import fl_model
        baseline_weights = deepcopy(self.global_model)
        mulinfos = []
        com_capa = 0
        for weight in weights:
            delta = 0
            for i, name in enumerate(weight):
                baseline = baseline_weights[name]
                delta += self.cal_mulinfo_one(weight[name], baseline, bins)
            mulinfos.append(delta)
        return mulinfos

    def cal_mulinfo_one(self, x, y, bins):
        c_x = np.histogram(x.asnumpy(), bins=bins)[0]
        c_y = np.histogram(y.asnumpy(), bins=bins)[0]
        c_xy = np.histogram2d(x.asnumpy().flatten(), y.asnumpy().flatten(), bins=bins)[0]

        c_x = c_x / np.sum(c_x)
        c_x = c_x[np.nonzero(c_x)]
        c_y = c_y / np.sum(c_y)
        c_y = c_y[np.nonzero(c_y)]
        c_xy = c_xy / np.sum(c_xy)
        c_xy = c_xy[np.nonzero(c_xy)]

        h_x = - sum(c_x * np.log2(c_x))
        h_y = - sum(c_y * np.log2(c_y))
        h_xy = -sum(c_xy * np.log2(c_xy))
        return h_x + h_y - h_xy

    def cal_proba(self, mulinfos):
        mulinfos = np.array(mulinfos)
        std = np.std(mulinfos)
        mulinfos = mulinfos - std
        thres = np.average(mulinfos)
        probas = np.zeros_like(mulinfos)
        probas[mulinfos < thres] = 1
        return probas
