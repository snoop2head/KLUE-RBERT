# RBERT for Relation Extraction task for KLUE

## Project Description

Relation Extraction task is one of [KLUE Benchmark](https://github.com/KLUE-benchmark/KLUE)'s task. 

Korean Language Understanding Evaluation(KLUE) Benchmark is composed of 8 tasks:

- Topic Classification (TC)
- Sentence Textual Similarity (STS)
- Natural Language Inference (NLI)
- Named Entity Recognition (NER)
- **Relation Extraction (RE)**
- (Part-Of-Speech) + Dependency Parsing (DP)
- Machine Reading Comprehension (MRC)
- Dialogue State Tracking (DST)

Relation extraction can be defined as multiclass classification task for relationship between subject entity and object entity.

Labels are such as `no_relation`, `per:employee_of`, `org:founded_by` totaling 30 labels. 

This repo contains custom dataset, custom training code utilizing [monologg's R-BERT Implementation](https://github.com/monologg/R-BERT).


## Arguments Usage

- RBERT

| Argument               | type  | Default                         | Explanation                                  |
| ---------------------- | ----- | ------------------------------- | -------------------------------------------- |
| batch_size             | int   | 40                              | batch size for training and inferece                |
| num_folds              | int   | 5                               | number of fold for Stratified KFold                 |
| num_train_epochs       | int   | 5                               | number of epochs for training                                   |
| loss                   | str   | focalloss                       | loss function                                |
| gamma                  | float | 1.0                             | focalloss's gamma value                    |
| optimizer              | str   | adamp                           | optimizer for training                               |
| scheduler              | str   | get_cosine_schedule_with_warmup | learning rate scheduler           |
| learning_rate          | float | 0.00005                         | initial learning rate                        |
| weight_decay           | float | 0.01                            | Loss function's weight decay, preventing overfit |
| warmup_step            | int   | 500                             |
| debug                  | bool  | false                           | debug with CPU device for better error representation                     |
| dropout_rate           | float | 0.1                             |                                  |
| save_steps             | int   | 100                             | number of steps for saving the model                            |
| evaluation_steps       | int   | 100                             | number of step until the evaluation                         |
| metric_for_best_model  | str   | eval/loss                       | the metric for determining which is the best model                  |
| load_best_model_at_end | bool  | True                            |

## References

- [monologg's R-BERT Implementation in Pytorch](https://github.com/monologg/R-BERT)
- [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284?context=cs)
- [KLUE: Korean Language Understanding Evaluation](https://github.com/KLUE-benchmark/KLUE)

## Authorship

- [jjonhwa](https://github.com/jjonhwa)
- [🤚 snoop2head](https://github.com/snoop2head)
- [kimyeondu](kimyeondu)
- [hihellohowareyou](https://github.com/hihellohowareyou)
- [shawnhyeonsoo](https://github.com/shawnhyeonsoo)
- [danielkim30433](https://github.com/danielkim30433)
- [ntommy11](https://github.com/ntommy11)

## Hardware

- `GPU : Tesla V100 32GB`
