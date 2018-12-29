## Shared Parameters 

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.engine.base_model.BaseModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |

## NaiveModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.naive_model.NaiveModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |

## DSSMModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.dssm_model.DSSMModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  | adam |  |
| with_multi_layer_perceptron |  | True |  |
| mlp_num_units |  |  |  |
| mlp_num_layers |  |  |  |
| mlp_num_fan_out |  |  |  |
| mlp_activation_func |  |  |  |

## CDSSMModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.cdssm_model.CDSSMModel'> |  |
| input_shapes |  | [(10, 900), (10, 900)] |  |
| task |  |  |  |
| optimizer |  | adam |  |
| with_multi_layer_perceptron |  | True |  |
| mlp_num_units |  |  |  |
| mlp_num_layers |  |  |  |
| mlp_num_fan_out |  |  |  |
| mlp_activation_func |  |  |  |
| w_initializer |  | glorot_normal |  |
| b_initializer |  | zeros |  |
| filters |  | 32 |  |
| kernel_size |  | 3 |  |
| strides |  | 1 |  |
| padding |  | same |  |
| conv_activation_func |  | tanh |  |

## DenseBaselineModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.dense_baseline_model.DenseBaselineModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |
| with_multi_layer_perceptron |  | True |  |
| mlp_num_units |  | 256 | quantitative uniform distribution in  [16, 512), with a step size of 1 |
| mlp_num_layers |  |  | quantitative uniform distribution in  [1, 5), with a step size of 1 |
| mlp_num_fan_out |  |  |  |
| mlp_activation_func |  |  |  |

## ArcIModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.arci_model.ArcIModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  | adam | choice in ['adam', 'rmsprop', 'adagrad'] |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| num_blocks |  | 1 |  |
| left_filters |  | [32] |  |
| left_kernel_sizes |  | [3] |  |
| right_filters |  | [32] |  |
| right_kernel_sizes |  | [3] |  |
| conv_activation_func |  | relu |  |
| left_pool_sizes |  | [2] |  |
| right_pool_sizes |  | [2] |  |
| padding |  | same | choice in ['same', 'valid', 'causal'] |
| dropout_rate |  |  | quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01 |

## ArcIIModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.arcii_model.ArcIIModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  | adam | choice in ['adam', 'rmsprop', 'adagrad'] |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| num_blocks |  | 1 |  |
| kernel_1d_count |  | 32 |  |
| kernel_1d_size |  | 3 |  |
| kernel_2d_count |  | [32] |  |
| kernel_2d_size |  | [[3, 3]] |  |
| activation |  | relu |  |
| pool_2d_size |  | [[2, 2]] |  |
| padding |  | same | choice in ['same', 'valid', 'causal'] |
| dropout_rate |  |  | quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01 |

## KNRMModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.knrm_model.KNRMModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| kernel_num |  | 11 | quantitative uniform distribution in  [5, 20), with a step size of 1 |
| sigma |  | 0.1 | quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01 |
| exact_sigma |  | 0.001 |  |

## DUETModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.duet_model.DUETModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| lm_filters |  | 32 |  |
| lm_hidden_sizes |  | [32] |  |
| dm_filters |  | 32 |  |
| dm_kernel_size |  | 3 |  |
| dm_q_hidden_size |  | 32 |  |
| dm_d_mpool |  | 3 |  |
| dm_hidden_sizes |  | [32] |  |
| lm_dropout_rate |  | 0.5 | quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.02 |
| dm_dropout_rate |  | 0.5 | quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.02 |

## DRMMTKSModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.drmmtks_model.DRMMTKSModel'> |  |
| input_shapes |  | [(5,), (300,)] |  |
| task |  |  |  |
| optimizer |  | adam |  |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| with_multi_layer_perceptron |  | True |  |
| mlp_num_units |  |  |  |
| mlp_num_layers |  |  |  |
| mlp_num_fan_out |  |  |  |
| mlp_activation_func |  |  |  |
| top_k |  | 10 | quantitative uniform distribution in  [2, 100), with a step size of 1 |

## DRMM

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.drmm.DRMM'> |  |
| input_shapes |  | [(5,), (5, 30)] |  |
| task |  |  |  |
| optimizer |  | adam |  |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| with_multi_layer_perceptron |  | True |  |
| mlp_num_units |  |  |  |
| mlp_num_layers |  |  |  |
| mlp_num_fan_out |  |  |  |
| mlp_activation_func |  |  |  |

## ANMMModel

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.anmm_model.ANMMModel'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| bin_num |  | 60 |  |
| dropout_rate |  | 0.1 |  |
| num_layers |  | 2 |  |
| hidden_sizes |  | [30, 30] |  |

## MatchLSTM

| name | description | default value | default hyper space |
|:--- |:--- |:--- |:--- |
| name |  |  |  |
| model_class | Model class. Used internally for save/load.  | <class 'matchzoo.models.match_lstm.MatchLSTM'> |  |
| input_shapes |  |  |  |
| task |  |  |  |
| optimizer |  |  |  |
| with_embedding |  | True |  |
| embedding_input_dim |  |  |  |
| embedding_output_dim |  |  |  |
| embedding_trainable |  |  |  |
| rnn_hidden_size |  | 256 | quantitative uniform distribution in  [128, 384), with a step size of 32 |
| fc_hidden_size |  | 200 | quantitative uniform distribution in  [100, 300), with a step size of 20 |

