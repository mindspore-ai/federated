import time
from src.deepfm import ModelBuilder, AUCMetric
from collections import defaultdict
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train.model import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from src.callback import EvalCallBack, LossCallBack
from copy import deepcopy

class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id, config, train_set, test_set):
        self.client_id = client_id
        self.config = self.download(config)
        self.train_set = train_set
        self.test_set = test_set
        self.model_builder = ModelBuilder(config, config)
        self.train_net, self.eval_net = self.model_builder.get_train_eval_net()
        self.auc_metric = AUCMetric()
        self.model = Model(self.train_net, eval_network=self.eval_net, metrics={"auc": self.auc_metric})
        self.time_callback = TimeMonitor(data_size=self.train_set.get_dataset_size())
        self.loss_callback = LossCallBack(loss_file_path=self.config.loss_file_name)
        self.callback_list = [self.time_callback, self.loss_callback]

        self.exec = False#设置当前参与者是否正在本地更新
        self.round_time = 0#参与fl的轮数
        self.pi = 0.1#记录当前的模型退化情况
        self.num_sele = 0
        self.num_agg = 0
        self.config = self.download(config)

    def local_execute(self, state_dict_to_load):
        print('Training on client #{}'.format(self.client_id))
        if state_dict_to_load is not None:
            load_param_into_net(self.train_net, state_dict_to_load)
        self.train_net.set_train()
        self.model.train(self.config.train_epochs, self.train_set, callbacks=self.callback_list)
        state_dict = deepcopy(self.train_net.parameters_dict())

        local_log = defaultdict(lambda: 0.0)
        if self.config.convert_dtype:
            self.config.convert_dtype = self.config.device_target != "CPU"
        load_param_into_net(self.eval_net, state_dict)
        self.eval_net.set_train(False)
        start = time.time()
        res = self.model.eval(self.test_set)
        eval_time = time.time() - start
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        res = list(res.values())[0].item()
        out_str = f'{time_str} AUC: {res}, eval time: {eval_time}s.'
        print(out_str)
        local_log['res'] = res

        self.local_result = {
            'state_dict': state_dict,
            'log': local_log
        }
    def set_round_time(self, t):
        self.round_time = t

    def set_exec(self):#改变节点的执行状态
        self.exec = not self.exec
    def set_agg(self):
        self.agg = not self.agg

    # Server interactions
    def download(self, argv):
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv