Vertical Federated Learning Communicator
===========================================

.. py:class:: VerticalFederatedCommunicator(http_server_config, remote_server_config)

    Define Vertical Federated Learning Communicator.

    Parameters：
        - **http_server_config** (ServerConfig) - Local Server Configuration.
        - **remote_server_config** (ServerConfig) - Peer server configuration.

    .. py:method:: launch()

        Start Vertical Federated Learning Communicator.

    .. py:method:: send_tensors(target_server_name, tensor_list_item_py)

        Send distributed training sensor data.

        Parameters：
            - **target_server_name** (str) - Specify the name of the peer server.
            - **tensor_list_item_py** (list[Tensor]) - Tensor Collection.

    .. py:method:: send_register(target_server_name, worker_register_item_py)

        Send worker registration message.

        Parameters：
            - **target_server_name** (str) - Specify the name of the peer server.
            - **worker_register_item_py** (str) - Worker registration information.

    .. py:method:: receive(target_server_name)

        Get the sensor data sent by the peer.

        Parameters：
            - **target_server_name** (str) - Specify the name of the peer server.

    .. py:method:: data_join_wait_for_start()

        Blocking and waiting for the registration information of the client worker.

    .. py:method:: http_server_config()

        Return local server configuration.

    .. py:method:: remote_server_config()

       Return to remote server configuration.

.. py:class:: ServerConfig(server_name, server_address)

    Define vertical federated server configuration.

    Parameters：
        - **server_name** (str) - Server name.
        - **server_address** (str) - server address.

    .. py:method:: init_server_config(http_server_config, remote_server_config)

        Initialize the local server configuration and remote server configuration, where remote server Config can be list[ServerConfig].

