import argparse
from mindspore_federated.data_join.worker import FLDataWorker


def get_parser():
    parser = argparse.ArgumentParser(description="Run run_data_join.py case")

    parser.add_argument("--role", type=str, default="leader")
    parser.add_argument("--worker_config_path", type=str, default="vfl/leader.yaml")
    parser.add_argument("--schema_path", type=str, default="vfl/schema.yaml")
    return parser


if __name__ == '__main__':
    args, _ = get_parser().parse_known_args()
    for key in args.__dict__:
        print('[', key, ']', args.__dict__[key], flush=True)
    role = args.role
    worker_config_path = args.worker_config_path
    schema_path = args.schema_path
    worker = FLDataWorker(role=role,
                          worker_config_path=worker_config_path,
                          data_schema_path=schema_path,
                          )
    worker.export()
