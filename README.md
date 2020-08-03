PyTorch-Based Distributed Knowledge Embedding (PDKE) framework is a distributed framework for Knowledge Graph Embedding. PDKE is developed on the basis of PyTorch-BigGraph (PBG) and implements four translation-based models: TransE, TransH, TransR, and TransD.

## Installation

PDKE is written in Python (version 3.6 or later) and relies on [PyTorch](https://pytorch.org/) (at least version 1.0) and a few other libraries. All computations are performed on the CPU, therefore a large number of cores is advisable. No GPU is necessary.

To install PDKE run:
```bash
git clone https://github.com/RweBs/PDKE.git
cd PDKE
python3 install setup.py
```

## Getting started

To perform training and evaluating, run

```bash
cd torchbiggraph/examples
python3 fb15k.py
```

Training will proceed for 30 epochs (which can be adjusted in the config file) in total, with the progress and some statistics logged to the console, for example:
```bash
Starting epoch 30 / 30, edge path 1 / 1, edge chunk 1 / 1
Edge path: data/FB15k/freebase_mtr100_mte100-train_partitioned
still in queue: 0
Swapping partitioned embeddings None ( 0 , 0 )
( 0 , 0 ): Loading entities 
( 0 , 0 ): bucket 1 / 1 : Processed 483142 edges in 580.29 s ( 0.00083 M/sec ); io: 0.02 s ( 545.83 MB/sec )
( 0 , 0 ): loss:  3.70139 , violators_lhs:  7.33727 , violators_rhs:  5.28217 , count:  483142
Swapping partitioned embeddings ( 0 , 0 ) NoneWriting partitioned embeddings
Finished epoch 30 / 30, edge path 1 / 1, edge chunk 1 / 1
```
After training, the evaluation process will be performed, and the evaluation results are as follows :
```
[Evaluator] Stats: pos_rank:  233.849 , mrr:  0.1018876 , r1:  0.00684769 , r10:  0.320522 , r50:  0.593904 , auc:  0.982623 , count:  59071
```

More information can be found in [PBG documentation](https://torchbiggraph.readthedocs.io/).
