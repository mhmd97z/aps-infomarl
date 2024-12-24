import yaml
import argparse
from distutils.util import strtobool
from argparse import Namespace


def aps_config(args, parser, yaml_path="./aps.yaml"):

    if yaml_path:
        with open(yaml_path, 'r') as file:
            yaml_config = yaml.safe_load(file)

        # Recursively convert YAML to Namespace
        def yaml_to_namespace(config):
            if isinstance(config, dict):
                return Namespace(**{key: yaml_to_namespace(value) for key, value in config.items()})
            return config

        yaml_namespace = yaml_to_namespace(yaml_config)



    parser.add_argument(
        "--embedding_size",
        type=int,
        default=16,
        help="Embedding layer output size for each category",
    )
    parser.add_argument(
        "--embed_hidden_size",
        type=int,
        default=16,
        help="Hidden layer dimension after the embedding layer",
    )
    parser.add_argument(
        "--embed_layer_N",
        type=int,
        default=1,
        help="Number of hidden linear layers after the " "embedding layer",
    )
    parser.add_argument(
        "--embed_use_ReLU",
        action="store_false",
        default=True,
        help="Whether to use ReLU in the linear layers after " "the embedding layer",
    )
    parser.add_argument(
        "--embed_add_self_loop",
        action="store_true",
        default=False,
        help="Whether to add self loops in adjacency matrix",
    )
    parser.add_argument(
        "--gnn_hidden_size",
        type=int,
        default=16,
        help="Hidden layer dimension in the GNN",
    )
    parser.add_argument(
        "--gnn_num_heads",
        type=int,
        default=3,
        help="Number of heads in the transformer conv layer (GNN)",
    )
    parser.add_argument(
        "--gnn_concat_heads",
        action="store_true",
        default=False,
        help="Whether to concatenate the head output or average",
    )
    parser.add_argument(
        "--gnn_layer_N", type=int, default=2, help="Number of GNN conv layers"
    )
    parser.add_argument(
        "--gnn_use_ReLU",
        action="store_false",
        default=True,
        help="Whether to use ReLU in GNN conv layers",
    )
    # parser.add_argument('--max_edge_dist', type=float, default=1,
    #                     help="Maximum distance above which edges cannot be "
    #                     "connected between the entities")
    parser.add_argument(
        "--graph_feat_type",
        type=str,
        default="global",
        choices=["global", "relative"],
        help="Whether to use " "'global' node/edge feats or 'relative'",
    )
    parser.add_argument(
        "--actor_graph_aggr",
        type=str,
        default="node",
        choices=["global", "node"],
        help="Whether we want to "
        "pull node specific features from the output or perform "
        "global_pool on all nodes. ",
    )
    parser.add_argument(
        "--critic_graph_aggr",
        type=str,
        default="global",
        choices=["global", "node"],
        help="Whether we want to "
        "pull node specific features from the output or perform "
        "global_pool on all nodes. ",
    )
    parser.add_argument(
        "--global_aggr_type",
        type=str,
        default="mean",
        choices=["mean", "max", "add"],
        help="The type of " "aggregation to perform if `graph_aggr` is `global`",
    )
    parser.add_argument(
        "--use_cent_obs",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Whether to use centralized observation " "for critic or not",
    )
    parser.add_argument(
        "--auto_mini_batch_size",
        action="store_true",
        default=False,
        help="Whether to automatically set mini batch size",
    )
    parser.add_argument(
        "--target_mini_batch_size",
        type=int,
        default=32,
        help="The target mini batch size to use",
    )

    all_args = parser.parse_known_args(args)[0]

    # Merge YAML Namespace into argparse Namespace
    def merge_namespaces(namespace, yaml_namespace):
        for key, value in vars(yaml_namespace).items():
            if isinstance(value, Namespace):
                if hasattr(namespace, key):
                    merge_namespaces(getattr(namespace, key), value)
                else:
                    setattr(namespace, key, value)
            else:
                setattr(namespace, key, value)

    merge_namespaces(all_args, yaml_namespace)
    
    if all_args.auto_mini_batch_size:
        # for recurrent generator only
        num_mini_batch = (
            all_args.n_rollout_threads * all_args.episode_length * all_args.num_agents
        ) // (all_args.target_mini_batch_size)
        new_batch_size = (
            (all_args.n_rollout_threads * all_args.episode_length * all_args.num_agents)
            // num_mini_batch
            // all_args.data_chunk_length
            * all_args.data_chunk_length
        )
        setattr(all_args, "num_mini_batch", num_mini_batch)
        print("_" * 50)
        print(f"Overriding num_mini_batch to {num_mini_batch}")
        print(f"Batch size to be: {new_batch_size}")
        print("_" * 50)
        
    return all_args, parser
