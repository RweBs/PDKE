#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():

    config = dict(
        # I/O data
        entity_path="data/FB15k",
        edge_paths=[
            "data/FB15k/freebase_mtr100_mte100-train_partitioned",
            "data/FB15k/freebase_mtr100_mte100-valid_partitioned",
            "data/FB15k/freebase_mtr100_mte100-test_partitioned",
        ],
        checkpoint_path="model/transE-20-bat200",

        # Graph structure
        entities={
            'all': {'num_partitions': 1},
        },
        relations=[{
            'name': 'all_edges',
            'lhs': 'all',
            'rhs': 'all',
            'operator': 'transE',
        }],
        dynamic_relations=True,

        # Scoring model
        relation_dimension=50,
        entity_dimension=50,
        global_emb=False,
        comparator='cos',

        # Training
        num_epochs=30,
        num_uniform_negs=200,
        loss_fn='ranking',
        batch_size=200,
        lr=0.05,

        # Evaluation during training
        eval_fraction=0,  # to reproduce results, we need to use all training data
    )

    return config
