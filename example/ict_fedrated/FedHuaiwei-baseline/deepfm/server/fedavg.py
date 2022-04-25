import numpy as np
import random
import xlwt
from src.dataset import create_dataset, DataType
from client import Client
from mindspore import Parameter
from copy import deepcopy

class FedAvg(object):

    def __init__(self, config):
        self.config = config
        self.global_model = None

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
            print('**************Round {}/{}******************'.format(round, rounds))
            train_log = self.train_round()

            print('Training finishes!')

            state_dict = deepcopy(train_log['state_dict'])
            self.global_model = deepcopy(state_dict)

            train_accuracy = train_log['log']['res']
            print('Accuracy is : {}'.format(train_accuracy))
            aggsheet.write(round, 0, round)
            aggsheet.write(round, 1, train_accuracy)
            aggsheet.write(round, 2, 0)
            aggsheet.write(round, 3, train_log['num_params'] * self.config.num_client_per_round)
            aggsheet.write(round, 4, self.config.num_client_per_round)
            aggsheet.write(round, 5, self.config.num_client_per_round)
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
        #用于做测试

        workbook.save(self.config.fl_mode+'_'+str(self.config.num_clients) + '_' + str(self.config.num_client_per_round) + '_.xls')
        print('Saved excel: '+ self.config.fl_mode+'_'+str(self.config.num_clients) + '_' + str(self.config.num_client_per_round) + '_.xls')

    def train_round(self):
        sample_clients = self.selection()
        local_results = []
        for i, client in enumerate(sample_clients):
            client.local_execute(state_dict_to_load=self.global_model)
            local_results.append(client.local_result)

        agg_local_results = self.aggregate_local_results(local_results)
        return agg_local_results

    def aggregate_local_results(self, local_results):
        num_parameters, state_dict = self.aggregate_local_state_dicts(
                [ltr['state_dict'] for ltr in local_results])
        return {
            'state_dict': state_dict,
            'num_params': num_parameters,
            'log': self.aggregate_local_logs(
                [ltr['log'] for ltr in local_results]
            )
        }


    def aggregate_local_state_dicts(self, local_train_state_dicts):
        num_parameters = 0
        agg_state_dict = {}
        for k in local_train_state_dicts[0]:#模型有几部分
            agg_state_dict[k] = 0#每部分初始值为0
            for lstd in local_train_state_dicts:#获取每一个client的本地模型
                agg_state_dict[k] += lstd[k]#对本地模型的每一部分作和
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

    def selection(self):
        clients_per_round = self.config.num_client_per_round
        sample_clients = [client for client in random.sample(
            self.clients, clients_per_round)]
        return sample_clients