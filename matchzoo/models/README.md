# MatchZoo Model Hyper-parameters Reference

## Shared Parameters 

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| name | Not related to the model's behavior. |  |  |
| model_class | Model class. Used internally for save/load. Changing this may cause unexpected behaviors. | <class 'matchzoo.engine.base_model.BaseModel'> |  |
| input_shapes | Dependent on the model and data. Should be set manually. |  |  |
| task | Decides model output shape, loss, and metrics. |  |  |
| optimizer |  |  | choice in ['adam', 'adgrad', 'rmsprop'] |

## NaiveModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |

## DSSMModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_multi_layer_perceptron | A flag of whether a multiple layer perceptron is used. Shouldn't be changed. | True |  |
| mlp_num_units | Number of units in first `mlp_num_layers` layers. |  |  |
| mlp_num_layers | Number of layers of the multiple layer percetron. |  |  |
| mlp_num_fan_out | Number of units of the layer that connects the multiple layer percetron and the output. |  |  |
| mlp_activation_func | Activation function used in the multiple layer perceptron. |  |  |

## CDSSMModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_multi_layer_perceptron | A flag of whether a multiple layer perceptron is used. Shouldn't be changed. | True |  |
| mlp_num_units | Number of units in first `mlp_num_layers` layers. |  |  |
| mlp_num_layers | Number of layers of the multiple layer percetron. |  |  |
| mlp_num_fan_out | Number of units of the layer that connects the multiple layer percetron and the output. |  |  |
| mlp_activation_func | Activation function used in the multiple layer perceptron. |  |  |
| filters |  | 32 |  |
| kernel_size |  | 3 |  |
| strides |  | 1 |  |
| padding |  | same |  |
| conv_activation_func |  | relu |  |
| w_initializer |  | glorot_normal |  |
| b_initializer |  | zeros |  |
| dropout_rate |  | 0.3 |  |

## DenseBaselineModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_multi_layer_perceptron | A flag of whether a multiple layer perceptron is used. Shouldn't be changed. | True |  |
| mlp_num_units | Number of units in first `mlp_num_layers` layers. | 256 | quantitative uniform distribution in  [16, 512), with a step size of 1 |
| mlp_num_layers | Number of layers of the multiple layer percetron. |  | quantitative uniform distribution in  [1, 5), with a step size of 1 |
| mlp_num_fan_out | Number of units of the layer that connects the multiple layer percetron and the output. |  |  |
| mlp_activation_func | Activation function used in the multiple layer perceptron. |  |  |

## ArcIModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
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
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
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
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
| kernel_num |  | 11 | quantitative uniform distribution in  [5, 20), with a step size of 1 |
| sigma |  | 0.1 | quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01 |
| exact_sigma |  | 0.001 |  |

## DUETModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
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
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
| with_multi_layer_perceptron | A flag of whether a multiple layer perceptron is used. Shouldn't be changed. | True |  |
| mlp_num_units | Number of units in first `mlp_num_layers` layers. |  |  |
| mlp_num_layers | Number of layers of the multiple layer percetron. |  |  |
| mlp_num_fan_out | Number of units of the layer that connects the multiple layer percetron and the output. |  |  |
| mlp_activation_func | Activation function used in the multiple layer perceptron. |  |  |
| top_k |  | 10 | quantitative uniform distribution in  [2, 100), with a step size of 1 |

## DRMM

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
| with_multi_layer_perceptron | A flag of whether a multiple layer perceptron is used. Shouldn't be changed. | True |  |
| mlp_num_units | Number of units in first `mlp_num_layers` layers. |  |  |
| mlp_num_layers | Number of layers of the multiple layer percetron. |  |  |
| mlp_num_fan_out | Number of units of the layer that connects the multiple layer percetron and the output. |  |  |
| mlp_activation_func | Activation function used in the multiple layer perceptron. |  |  |

## ANMMModel

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
| bin_num |  | 60 |  |
| dropout_rate |  | 0.1 |  |
| num_layers |  | 2 |  |
| hidden_sizes |  | [30, 30] |  |

## MatchLSTM

| name | description | default value | default hyper space |
| ---:|:--- |:--- |:--- |
| with_embedding | A flag used help `auto` module. Shouldn't be changed. | True |  |
| embedding_input_dim | Usually equals vocab size + 1. Should be set manually. |  |  |
| embedding_output_dim | Should be set manually. |  |  |
| embedding_trainable | `True` to enable embedding layer training, `False` to freeze embedding parameters. |  |  |
| rnn_hidden_size |  | 256 | quantitative uniform distribution in  [128, 384), with a step size of 32 |
| fc_hidden_size |  | 200 | quantitative uniform distribution in  [100, 300), with a step size of 20 |

