# Language Generation with Multi-hop Reasoning on Commonsense Knowledge Graph

## Introduction

This is the pytorch implementation of our paper "*Language Generation with Multi-hop Reasoning on Commonsense Knowledge Graph*". 
The arxiv version of the paper could be found [here](https://arxiv.org/pdf/2009.11692.pdf).


## Requirements

```
python version >= 3
torch version >= 1.4.0
transformers == 2.8.0
nltk == 3.4.5
networkx == 2.1
spacy == 2.2.1
torch-scatter == 2.0.5+${CUDA}
```

For `torch-scatter`, `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation. 
For more information check [here](https://github.com/rusty1s/pytorch_scatter).


## Preprocessing

Preprocessed datasets can be downloaded from [here](https://drive.google.com/file/d/15ckbKsyq5sMU-RJh0n-mB9NgyW1WhTsF/view?usp=sharing).

Unzip the file and move it to `data`.

Extract English ConceptNet and build graph.

```bash
cd data
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../preprocess
python extract_cpnet.py
python graph_construction.py
```

Preprocessing multi-hop relational paths for the model. Set `$DATA` to either `anlg`, `eg`, `story`.

```bash
export DATA=eg
python ground_concepts_simple.py $DATA
python find_neighbours.py $DATA
python filter_triple.py $DATA
```

Download the pre-trained GPT-2 model.

```bash
mkdir -p models
cd models
mkdir -p gpt2-small
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-pytorch_model.bin
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json
```

Add special tokens to vocabulary.

```bash
cd scripts
python add_special_tokens.py
```

## Usage

### Training 

The following command is an example to train the model on the trarining set and evaluate on the development set. Set `$DATA_TYPE` to either `anlg`, `eg`, `story`.

```bash
export DATA_TYPE={anlg, eg, story}
export ROOT_PATH=..
export DEVICE=1
CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE} \
--source_length 32 \
--target_length 32 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/gpt2-small \
--do_train \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--num_train_epochs 3 \
--learning_rate 1e-5 \
--aggregate_method max \
--alpha 3 \
--beta 5 \
--gamma 0.5 \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--logging_steps 20 \
```

**Note:** Some scatter operations in `torch_scatter` are currently non-deterministic and not controlled by the random seed due to the usage of atomic operations. 
See the discussions [here](https://discuss.pytorch.org/t/possible-solution-to-the-reproducibility-issues-using-scatter-add-operation/48989). 
So different runs may vary slightly. 

To reproduce our results, you can directly download our [checkpoints](https://drive.google.com/drive/folders/1yx6WTjR1mXYZ6W7Pd-_MI7XkWfyV78c9?usp=sharing) and evaluate the model.

### Inference

The following command is an example to run the inference on the test set. Set `$DATA_TYPE` to either `anlg`, `eg`, `story`.

```bash
export DATA_TYPE={anlg, eg, story}
export ROOT_PATH=..
export DEVICE=1
CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE} \
--source_length 32 \
--target_length 32 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/gpt2-small \
--do_eval \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--aggregate_method max \
--gamma 0.5 \
```

We also provide generated results in `results/` on the three tasks and you can directly evaluate the results with our evaluation scripts.

### Evaluation

To evaluate the generated results on `anlg` and `eg`, run the following commands. Set `$DATA` to either `anlg`, `eg`.

```bash
export DATA={eg, anlg}
python evaluation/eval.py --dataset ${DATA} --output_dir results/grf-${DATA}.txt
```

To evaluate the generated results on `story`, run the following commands.

```bash
python eval_story.py results/grf-story.txt
```

## Citation

```
@inproceedings{ji2020language,
    title = "Language Generation with Multi-Hop Reasoning on Commonsense Knowledge Graph",
    author = "Ji, Haozhe and Ke, Pei and Huang, Shaohan and Wei, Furu and Zhu, Xiaoyan and Huang, Minlie",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    year = "2020",
}
```

**Please kindly cite our paper if you find this paper and the codes helpful.**

## Thanks

Many thanks to the Github repository of [Transformers](https://github.com/huggingface/transformers), [fairseq](https://github.com/pytorch/fairseq) and [KagNet](https://github.com/INK-USC/KagNet). Part of our codes are modified based on their codes.
