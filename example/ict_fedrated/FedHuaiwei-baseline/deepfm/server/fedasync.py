import numpy as np
import random
import datetime

from src.dataset import create_dataset, DataType
from client import Client
from mindspore import Parameter
from copy import deepcopy
import xlwt

workbook = xlwt.Workbook(encoding='ascii')
worksheet_sele = workbook.add_sheet('selection')
worksheet_agg = workbook.add_sheet('aggregation')
worksheet_client = workbook.add_sheet('client')


class FedAsync(object):
    """Basic federated learning server."""
    def __init__(self, config):#向服务器传入配置文件作为参数
        self.config = config
        self.global_model = None
        self.agg_id = []#id
        self.client_training_time = []#time
        self.time_of_round = 0 #上一轮的一轮fl的时间
        self.client_of_this_round = []#本轮选择的客户端数量

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
        all_size = dataset_all.get_dataset_size() #41300

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

    def run(self):
        worksheet_agg.write(0, 0, label='聚合轮次')
        worksheet_agg.write(1, 0, label='选择节点数')
        worksheet_agg.write(2, 0, label='accuracy')
        worksheet_agg.write(3, 0, label='节点编号')

        rounds = self.config.max_round
        print('total fl round is: {}'.format(rounds))
        for round in range(rounds):
            self.select_client(round)
            self.aggregate(round)

        worksheet_client.write(0, 0, label='id')
        worksheet_client.write(0, 1, label='num_sele')
        worksheet_client.write(0, 2, label='num_agg')
        for client in self.clients:
            worksheet_client.write(client.client_id + 1, 0, client.client_id)
            worksheet_client.write(client.client_id + 1, 1, client.num_sele)
            worksheet_client.write(client.client_id + 1, 2, client.num_agg)

        workbook.save(self.config.fl_mode  + '.xls')
        print('Saved excel: ' + self.config.fl_mode  + '.xls')

    def select_client(self, round):
        worksheet_sele.write(0, round, label='第{}轮'.format(round))
        sample_clients = self.selection(round)
        idx = 1
        starttime = datetime.datetime.now()
        self.client_of_this_round.clear()
        for client in sample_clients:
            worksheet_sele.write(idx, round, client.client_id)  # 写入每轮选择的参与者的编号
            idx = idx + 1
            client.set_exec()
            client.set_round_time(round)
            client.num_sele += 1
            client.local_execute(self.global_model)
            self.agg_id.append(client.client_id)
            self.client_of_this_round.append(client.client_id)

        endtime = datetime.datetime.now()
        if round == 0:
            avg_time = (endtime - starttime).seconds / len(sample_clients)
            self.config.sigma = avg_time / 5
            self.times = np.random.normal(avg_time, self.config.sigma, self.config.num_clients)
            print(self.times)
        for client in sample_clients:
             self.client_training_time.append(self.times[client.client_id])
        print(self.times)

    def aggregate(self, round):

        worksheet_agg.write(0, round + 1, label='第{}轮'.format(round + 1))
        print('第#{}次聚合'.format(round + 1))

        print('每轮训练时间: {}'.format(self.time_of_round))

        for i in range(len(self.agg_id)):
            if self.clients[self.agg_id[i]].round_time < round:
                self.client_training_time[i] -= self.time_of_round

        num_select_clients = self.config.agg_client_per_round
        print('聚合数目', num_select_clients)

        agg_clients = []

        while num_select_clients > 0:
            training_time = min(self.client_training_time)
            if num_select_clients == 1:
                if training_time > 0:
                    self.time_of_round = training_time
                else:
                    self.time_of_round = 0
            idx = self.client_training_time.index(training_time)
            client_id = self.agg_id[idx]
            agg_clients.append(self.clients[client_id])
            self.agg_id.remove(client_id)
            self.client_training_time.remove(training_time)
            num_select_clients -= 1
        idx = 1
        worksheet_agg.write(idx, round + 1, len(agg_clients))  # 先记录聚合的节点数量
        idx = idx + 2

        local_results = []
        local_pi = []
        for client in agg_clients:  # 更新聚合状态
            print('聚合的client的id为：', client.client_id)
            worksheet_agg.write(idx, round + 1, client.client_id)  # 最后记录参与的节点编号
            idx = idx + 1
            self.clients[client.client_id].set_exec()
            self.clients[client.client_id].num_agg += 1
            local_results.append(client.local_result)
            client.pi = self.Calculate_Staleness(client.round_time, round)
            local_pi.append(client.pi)

        print('Aggregating updates...')
        starttime = datetime.datetime.now()
        agg_results = self.aggregate_local_results(local_results, local_pi)
        endtime = datetime.datetime.now()
        self.global_model = deepcopy(agg_results['state_dict'])
        accuracy = agg_results['log']['res']
        print('Average accuracy: {:.2f}%\n'.format(100 * accuracy))
        self.time_of_round += (endtime - starttime).seconds
        worksheet_agg.write(idx - len(agg_clients) - 1, round + 1, accuracy)  # 第二记录每轮聚合的精度
        worksheet_agg.write(31, round + 1, (endtime - starttime).seconds)

    def aggregate_local_results(self, local_results, local_pi):
        state_dict = self.aggregate_local_state_dicts(
            [ltr['state_dict'] for ltr in local_results], local_pi)
        return {
            'state_dict': state_dict,
            'log': self.aggregate_local_logs(
                [ltr['log'] for ltr in local_results]
            )}

    def aggregate_local_state_dicts(self, local_train_state_dicts, local_pi):
        agg_state_dict = {}
        total = len(local_train_state_dicts)
        for idx, k in enumerate(local_train_state_dicts[0]):
            agg_state_dict[k] = 0
            for i, lstd in enumerate(local_train_state_dicts):
                agg_state_dict[k] += lstd[k] * local_pi[i]
            agg_state_dict[k] /= total
            agg_state_dict[k] = Parameter(default_input=agg_state_dict[k], name=k)
        return agg_state_dict

    def aggregate_local_logs(self, local_logs):
        agg_log = deepcopy(local_logs[0])
        for k in agg_log:
            agg_log[k] = 0
            for local_log_idx, local_log in enumerate(local_logs):
                agg_log[k] += local_log[k] * self.config.num_samples
        for k in agg_log:
            agg_log[k] /= self.config.num_samples * len(local_logs)
        return agg_log

    def Calculate_Staleness(self, client_round, agg_round):
        if self.config.cal_staleness == 1:  # async退化情况计算
            a = 10
            b = 4
            if agg_round - client_round <= b:
                return 1
            else:
                return 1.0 / (a * (agg_round - client_round - b) + 1)
        else:
            a = 0.5
            return 1.0 / ((agg_round - client_round + 0.999999) ** a)

    def selection(self, round):
        clients_per_round = self.config.num_client_per_round
        client_set = []
        for client in self.clients:
            if not client.exec:
                client_set.append(client)

        if len(client_set) < clients_per_round:
            clients_per_round = len(client_set)
        sample_clients = [client for client in random.sample(
            client_set, clients_per_round)]
        return sample_clients


