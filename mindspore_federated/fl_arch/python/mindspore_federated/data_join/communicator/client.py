from mindspore_federated._mindspore_federated import RunPSI


class _DataJoinClient:
    def __init__(self, worker_config):
        """
        Data join client.

        Args:
            worker_config (_WorkerConfig): The config of worker.
        """
        self._worker_config = worker_config

    def wait_for_negotiated(self):
        """
        Negotiate hyper parameters with server.
        """
        return self._request_hyper_params()

    def _request_hyper_params(self):
        """
        Request and verify hyper parameters from server. Overwrite hyper parameters in local worker config.

        The hyper parameters include:
            primary_key (str)
            bin_num (int)
            shard_num (int)
            join_type (str)

        Returns:
            - worker_config (_WorkerConfig): The config of worker.
        """
        # TODO: send the above hyper parameters to client
        # http_server_address = self._worker_config.http_server_address
        # remote_server_address = self._worker_config.remote_server_address
        # worker_config = request_params(http_server_address, remote_server_address)
        # self._worker_config.primary_key = worker_config.primary_key
        # self._worker_config.bin_num = worker_config.bin_num
        # self._worker_config.shard_num = worker_config.shard_num
        # self._worker_config.join_type = worker_config.join_type
        return self._worker_config

    def join_func(self, input_vct, bin_id):
        """
        Join function.

        Args:
            input_vct (list(str)): The keys need to be joined. The type of each key must be "str".
            bin_id (int): The id of the bin.

        Returns:
            - intersection_keys (list(str)): The intersection keys.

        Raises:
            ValueError: If the join type is not supported.
        """
        if self._worker_config.join_type == "psi":
            # TODO: real psi
            # thread_num = self._worker_config.thread_num
            # http_server_address = self._worker_config.http_server_address
            # remote_server_address = self._worker_config.remote_server_address
            # intersection_keys = RunPSI(input_vct, "client", http_server_address, remote_server_address,
            #                            thread_num, bin_id)
            # return intersection_keys
            return input_vct[:666]
        raise ValueError("join type: {} is not support currently".format(self._worker_config.join_type))
