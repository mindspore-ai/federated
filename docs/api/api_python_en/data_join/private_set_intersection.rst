Private Set Intersection
================================

.. py:function:: RunPSI(input_data, comm_role, peer_comm_role, bucket_id, thread_num)

    Private set intersection protocol.

    .. note::
        Use `from mindspore_federated._mindspore_federated import RunPSI` to import this interface;
        A vertical federated communication instance must be initialized before calling this interface. See `MindSpore federated ST <https://gitee.com/mindspore/federated/blob/master/tests/st/psi/run_psi.py>`_ .

    Parameters:
        - **input_data** (list[string]) - Self input dataset.
        - **comm_role** (string) - Self communication role, "server" or "client".
        - **peer_comm_role** (string) - The peer communication role, "server" or "client".
        - **bucket_id** (int) - Bucket index from the external bucket division. During the running of the protocol, the parties' bucket index must be consistent, otherwise the server will abort and the client will be blocked.
        - **thread_num** (int) - Thread number. Set to 0 means the maximum available thread number of the machine minus 5. The final value will be restrict to the range of 1 to the maximum available thread number.

    Return:
        - **result** (list[string]) - The intersection set.

    Exception:
        - **TypeError** - The input type of `input_data` is not list[string].
        - **TypeError** - The input type of `bucket_id` is not a integer larger than or equal to 0, such as a negative or decimal number.
        - **TypeError** - The input type of `thread_num` is not a integer larger than or equal to 0, such as a negative or decimal number.


.. py:function:: PlainIntersection(input_data, comm_role, peer_comm_role, bucket_id, thread_num)

    Plain set intersection protocol.

    .. note::
        Use `from mindspore_federated._mindspore_federated import PlainIntersection` to import this interface;
        A vertical federated communication instance must be initialized before calling this interface. See `MindSpore federated ST <https://gitee.com/mindspore/federated/blob/master/tests/st/psi/run_psi.py>`_ .

    Parameters:
        - **input_data** (list[string]) - Self input dataset.
        - **comm_role** (string) - Self communication role, "server" or "client".
        - **peer_comm_role** (string) - The peer communication role, "server" or "client".
        - **bucket_id** (int) - Bucket index from the external bucket division. During the running of the protocol, the parties' bucket index must be consistent, otherwise the server will abort and the client will be blocked.
        - **thread_num** (int) - Thread number. Set to 0 means the maximum available thread number of the machine minus 5. The final value will be restrict to the range of 1 to the maximum available thread number.

    Return:
        - **result** (list[string]) - The intersection set.

    Exception:
        - **TypeError** - The input type of `input_data` is not list[string].
        - **TypeError** - The input type of `bucket_id` is not a integer larger than or equal to 0, such as a negative or decimal number.
        - **TypeError** - The input type of `thread_num` is not a integer larger than or equal to 0, such as a negative or decimal number.