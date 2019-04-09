************************
MatchZoo Model Reference
************************

Naive
#####

Model Documentation
*******************

Naive model with a simplest structure for testing purposes.

Bare minimum functioning model. The best choice to get things rolling.
The worst choice to fit and evaluate performance.

Model Hyper Parameters
**********************

====  ============  =========================================================================================  =====================================  ========================================
  ..  Name          Description                                                                                Default Value                          Default Hyper-Space
====  ============  =========================================================================================  =====================================  ========================================
   0  model_class   Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.naive.Naive'>
   1  input_shapes  Dependent on the model and data. Should be set manually.
   2  task          Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                adam                                   choice in ['adam', 'adagrad', 'rmsprop']
====  ============  =========================================================================================  =====================================  ========================================

DSSM
####

Model Documentation
*******************

Deep structured semantic model.

Examples:
    >>> model = DSSM()
    >>> model.params['mlp_num_layers'] = 3
    >>> model.params['mlp_num_units'] = 300
    >>> model.params['mlp_num_fan_out'] = 128
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  ===================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================  ===================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.dssm.DSSM'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                               adam
   4  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   5  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
   6  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
   7  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
   8  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
====  ===========================  =========================================================================================  ===================================  =====================================================================

CDSSM
#####

Model Documentation
*******************

CDSSM Model implementation.

Learning Semantic Representations Using Convolutional Neural Networks
for Web Search. (2014a)
A Latent Semantic Model with Convolutional-Pooling Structure for
Information Retrieval. (2014b)

Examples:
    >>> model = CDSSM()
    >>> model.params['optimizer'] = 'adam'
    >>> model.params['filters'] =  32
    >>> model.params['kernel_size'] = 3
    >>> model.params['conv_activation_func'] = 'relu'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =============================================================================================  =====================================  =====================================================================
  ..  Name                         Description                                                                                    Default Value                          Default Hyper-Space
====  ===========================  =============================================================================================  =====================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.      <class 'matchzoo.models.cdssm.CDSSM'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                                   adam
   4  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                   True
   5  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                              128                                    quantitative uniform distribution in  [8, 256), with a step size of 8
   6  mlp_num_layers               Number of layers of the multiple layer percetron.                                              3                                      quantitative uniform distribution in  [1, 6), with a step size of 1
   7  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.        64                                     quantitative uniform distribution in  [4, 128), with a step size of 4
   8  mlp_activation_func          Activation function used in the multiple layer perceptron.                                     relu
   9  filters                      Number of filters in the 1D convolution layer.                                                 32
  10  kernel_size                  Number of kernel size in the 1D convolution layer.                                             3
  11  strides                      Strides in the 1D convolution layer.                                                           1
  12  padding                      The padding mode in the convolution layer. It should be one of `same`, `valid`, and `causal`.  same
  13  conv_activation_func         Activation function in the convolution layer.                                                  relu
  14  w_initializer                                                                                                               glorot_normal
  15  b_initializer                                                                                                               zeros
  16  dropout_rate                 The dropout rate.                                                                              0.3
====  ===========================  =============================================================================================  =====================================  =====================================================================

DenseBaseline
#############

Model Documentation
*******************

A simple densely connected baseline model.

Examples:
    >>> model = DenseBaseline()
    >>> model.params['mlp_num_layers'] = 2
    >>> model.params['mlp_num_units'] = 300
    >>> model.params['mlp_num_fan_out'] = 128
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()
    >>> model.compile()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  ======================================================  ======================================================================
  ..  Name                         Description                                                                                Default Value                                           Default Hyper-Space
====  ===========================  =========================================================================================  ======================================================  ======================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.dense_baseline.DenseBaseline'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                               adam
   4  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   5  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          256                                                     quantitative uniform distribution in  [16, 512), with a step size of 1
   6  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                                       quantitative uniform distribution in  [1, 5), with a step size of 1
   7  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                                      quantitative uniform distribution in  [4, 128), with a step size of 4
   8  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
====  ===========================  =========================================================================================  ======================================================  ======================================================================

ArcI
####

Model Documentation
*******************

ArcI Model.

Examples:
    >>> model = ArcI()
    >>> model.params['num_blocks'] = 1
    >>> model.params['left_filters'] = [32]
    >>> model.params['right_filters'] = [32]
    >>> model.params['left_kernel_sizes'] = [3]
    >>> model.params['right_kernel_sizes'] = [3]
    >>> model.params['left_pool_sizes'] = [2]
    >>> model.params['right_pool_sizes'] = [4]
    >>> model.params['conv_activation_func'] = 'relu'
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 64
    >>> model.params['mlp_num_fan_out'] = 32
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.params['dropout_rate'] = 0.5
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  ============================================================================================  ===================================  ==========================================================================
  ..  Name                         Description                                                                                   Default Value                        Default Hyper-Space
====  ===========================  ============================================================================================  ===================================  ==========================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.     <class 'matchzoo.models.arci.ArcI'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                                  adam
   4  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                         True
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  embedding_trainable          `True` to enable embedding layer training, `False` to freeze embedding parameters.            True
   8  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.                  True
   9  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                             128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
  10  mlp_num_layers               Number of layers of the multiple layer percetron.                                             3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  11  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.       64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
  12  mlp_activation_func          Activation function used in the multiple layer perceptron.                                    relu
  13  num_blocks                   Number of convolution blocks.                                                                 1
  14  left_filters                 The filter size of each convolution blocks for the left input.                                [32]
  15  left_kernel_sizes            The kernel size of each convolution blocks for the left input.                                [3]
  16  right_filters                The filter size of each convolution blocks for the right input.                               [32]
  17  right_kernel_sizes           The kernel size of each convolution blocks for the right input.                               [3]
  18  conv_activation_func         The activation function in the convolution layer.                                             relu
  19  left_pool_sizes              The pooling size of each convolution blocks for the left input.                               [2]
  20  right_pool_sizes             The pooling size of each convolution blocks for the right input.                              [2]
  21  padding                      The padding mode in the convolution layer. It should be oneof `same`, `valid`, and `causal`.  same                                 choice in ['same', 'valid', 'causal']
  22  dropout_rate                 The dropout rate.                                                                             0.0                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ===========================  ============================================================================================  ===================================  ==========================================================================

ArcII
#####

Model Documentation
*******************

ArcII Model.

Examples:
    >>> model = ArcII()
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['num_blocks'] = 2
    >>> model.params['kernel_1d_count'] = 32
    >>> model.params['kernel_1d_size'] = 3
    >>> model.params['kernel_2d_count'] = [16, 32]
    >>> model.params['kernel_2d_size'] = [[3, 3], [3, 3]]
    >>> model.params['pool_2d_size'] = [[2, 2], [2, 2]]
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================  =====================================  ==========================================================================
  ..  Name                  Description                                                                                Default Value                          Default Hyper-Space
====  ====================  =========================================================================================  =====================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.arcii.ArcII'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                        adam                                   choice in ['adam', 'rmsprop', 'adagrad']
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  num_blocks            Number of 2D convolution blocks.                                                           1
   9  kernel_1d_count       Kernel count of 1D convolution layer.                                                      32
  10  kernel_1d_size        Kernel size of 1D convolution layer.                                                       3
  11  kernel_2d_count       Kernel count of 2D convolution layer ineach block                                          [32]
  12  kernel_2d_size        Kernel size of 2D convolution layer in each block.                                         [[3, 3]]
  13  activation            Activation function.                                                                       relu
  14  pool_2d_size          Size of pooling layer in each block.                                                       [[2, 2]]
  15  padding               The padding mode in the convolution layer. It should be oneof `same`, `valid`.             same                                   choice in ['same', 'valid']
  16  dropout_rate          The dropout rate.                                                                          0.0                                    quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================  =====================================  ==========================================================================

MatchPyramid
############

Model Documentation
*******************

MatchPyramid Model.

Examples:
    >>> model = MatchPyramid()
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['num_blocks'] = 2
    >>> model.params['kernel_count'] = [16, 32]
    >>> model.params['kernel_size'] = [[3, 3], [3, 3]]
    >>> model.params['dpool_size'] = [3, 10]
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================  ====================================================  ==========================================================================
  ..  Name                  Description                                                                                Default Value                                         Default Hyper-Space
====  ====================  =========================================================================================  ====================================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.match_pyramid.MatchPyramid'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                        adam
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  num_blocks            Number of convolution blocks.                                                              1
   9  kernel_count          The kernel count of the 2D convolution of each block.                                      [32]
  10  kernel_size           The kernel size of the 2D convolution of each block.                                       [[3, 3]]
  11  activation            The activation function.                                                                   relu
  12  dpool_size            The max-pooling size of each block.                                                        [3, 10]
  13  padding               The padding mode in the convolution layer.                                                 same
  14  dropout_rate          The dropout rate.                                                                          0.0                                                   quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.01
====  ====================  =========================================================================================  ====================================================  ==========================================================================

KNRM
####

Model Documentation
*******************

KNRM model.

Examples:
    >>> model = KNRM()
    >>> model.params['embedding_input_dim'] =  10000
    >>> model.params['embedding_output_dim'] =  10
    >>> model.params['embedding_trainable'] = True
    >>> model.params['kernel_num'] = 11
    >>> model.params['sigma'] = 0.1
    >>> model.params['exact_sigma'] = 0.001
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================  ===================================  ===========================================================================
  ..  Name                  Description                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================  ===================================  ===========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.knrm.KNRM'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                        adam
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  kernel_num            The number of RBF kernels.                                                                 11                                   quantitative uniform distribution in  [5, 20), with a step size of 1
   9  sigma                 The `sigma` defines the kernel width.                                                      0.1                                  quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01
  10  exact_sigma           The `exact_sigma` denotes the `sigma` for exact match.                                     0.001
====  ====================  =========================================================================================  ===================================  ===========================================================================

DUET
####

Model Documentation
*******************

DUET Model.

Examples:
    >>> model = DUET()
    >>> model.params['embedding_input_dim'] = 1000
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['lm_filters'] = 32
    >>> model.params['lm_hidden_sizes'] = [64, 32]
    >>> model.params['dropout_rate'] = 0.5
    >>> model.params['dm_filters'] = 32
    >>> model.params['dm_kernel_size'] = 3
    >>> model.params['dm_d_mpool'] = 4
    >>> model.params['dm_hidden_sizes'] = [64, 32]
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =============================================================================================  ===================================  ==========================================================================
  ..  Name                  Description                                                                                    Default Value                        Default Hyper-Space
====  ====================  =============================================================================================  ===================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.      <class 'matchzoo.models.duet.DUET'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                            adam
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                          True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.             True
   8  lm_filters            Filter size of 1D convolution layer in the local model.                                        32
   9  lm_hidden_sizes       A list of hidden size of the MLP layer in the local model.                                     [32]
  10  dm_filters            Filter size of 1D convolution layer in the distributed model.                                  32
  11  dm_kernel_size        Kernel size of 1D convolution layer in the distributed model.                                  3
  12  dm_q_hidden_size      Hidden size of the MLP layer for the left text in the distributed model.                       32
  13  dm_d_mpool            Max pooling size for the right text in the distributed model.                                  3
  14  dm_hidden_sizes       A list of hidden size of the MLP layer in the distributed model.                               [32]
  15  padding               The padding mode in the convolution layer. It should be one of `same`, `valid`, and `causal`.  same
  16  activation_func       Activation function in the convolution layer.                                                  relu
  17  dropout_rate          The dropout rate.                                                                              0.5                                  quantitative uniform distribution in  [0.0, 0.8), with a step size of 0.02
====  ====================  =============================================================================================  ===================================  ==========================================================================

DRMMTKS
#######

Model Documentation
*******************

DRMMTKS Model.

Examples:
    >>> model = DRMMTKS()
    >>> model.params['embedding_input_dim'] = 10000
    >>> model.params['embedding_output_dim'] = 100
    >>> model.params['top_k'] = 20
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 5
    >>> model.params['mlp_num_fan_out'] = 1
    >>> model.params['mlp_activation_func'] = 'tanh'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  =========================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                              Default Hyper-Space
====  ===========================  =========================================================================================  =========================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.drmmtks.DRMMTKS'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.                                   [(5,), (300,)]
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                               adam
   4  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  embedding_trainable          `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   9  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                        quantitative uniform distribution in  [8, 256), with a step size of 8
  10  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                          quantitative uniform distribution in  [1, 6), with a step size of 1
  11  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                         quantitative uniform distribution in  [4, 128), with a step size of 4
  12  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  13  mask_value                   The value to be masked from inputs.                                                        -1
  14  top_k                        Size of top-k pooling layer.                                                               10                                         quantitative uniform distribution in  [2, 100), with a step size of 1
====  ===========================  =========================================================================================  =========================================  =====================================================================

DRMM
####

Model Documentation
*******************

DRMM Model.

Examples:
    >>> model = DRMM()
    >>> model.params['mlp_num_layers'] = 1
    >>> model.params['mlp_num_units'] = 5
    >>> model.params['mlp_num_fan_out'] = 1
    >>> model.params['mlp_activation_func'] = 'tanh'
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()
    >>> model.compile()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  ===================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                        Default Hyper-Space
====  ===========================  =========================================================================================  ===================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.drmm.DRMM'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.                                   [(5,), (5, 30)]
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                               adam
   4  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  embedding_trainable          `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   9  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                  quantitative uniform distribution in  [8, 256), with a step size of 8
  10  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                    quantitative uniform distribution in  [1, 6), with a step size of 1
  11  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                   quantitative uniform distribution in  [4, 128), with a step size of 4
  12  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  13  mask_value                   The value to be masked from inputs.                                                        -1
====  ===========================  =========================================================================================  ===================================  =====================================================================

ANMM
####

Model Documentation
*******************

ANMM Model.

Examples:
    >>> model = ANMM()
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================  ===================================  ======================================================================
  ..  Name                  Description                                                                                Default Value                        Default Hyper-Space
====  ====================  =========================================================================================  ===================================  ======================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.anmm.ANMM'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                        adam
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  dropout_rate          The dropout rate.                                                                          0.1                                  quantitative uniform distribution in  [0, 1), with a step size of 0.05
   9  num_layers            Number of hidden layers in the MLP layer.                                                  2
  10  hidden_sizes          Number of hidden size for each hidden layer                                                [30, 30]
====  ====================  =========================================================================================  ===================================  ======================================================================

MVLSTM
######

Model Documentation
*******************

MVLSTM Model.

Examples:
    >>> model = MVLSTM()
    >>> model.params['lstm_units'] = 32
    >>> model.params['top_k'] = 50
    >>> model.params['mlp_num_layers'] = 2
    >>> model.params['mlp_num_units'] = 20
    >>> model.params['mlp_num_fan_out'] = 10
    >>> model.params['mlp_activation_func'] = 'relu'
    >>> model.params['dropout_rate'] = 0.5
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ===========================  =========================================================================================  =======================================  =====================================================================
  ..  Name                         Description                                                                                Default Value                            Default Hyper-Space
====  ===========================  =========================================================================================  =======================================  =====================================================================
   0  model_class                  Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.mvlstm.MVLSTM'>
   1  input_shapes                 Dependent on the model and data. Should be set manually.
   2  task                         Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                               adam
   4  with_embedding               A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim          Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim         Should be set manually.
   7  embedding_trainable          `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  with_multi_layer_perceptron  A flag of whether a multiple layer perceptron is used. Shouldn't be changed.               True
   9  mlp_num_units                Number of units in first `mlp_num_layers` layers.                                          128                                      quantitative uniform distribution in  [8, 256), with a step size of 8
  10  mlp_num_layers               Number of layers of the multiple layer percetron.                                          3                                        quantitative uniform distribution in  [1, 6), with a step size of 1
  11  mlp_num_fan_out              Number of units of the layer that connects the multiple layer percetron and the output.    64                                       quantitative uniform distribution in  [4, 128), with a step size of 4
  12  mlp_activation_func          Activation function used in the multiple layer perceptron.                                 relu
  13  lstm_units                   Integer, the hidden size in the bi-directional LSTM layer.                                 32
  14  dropout_rate                 Float, the dropout rate.                                                                   0.0
  15  top_k                        Integer, the size of top-k pooling layer.                                                  10                                       quantitative uniform distribution in  [2, 100), with a step size of 1
====  ===========================  =========================================================================================  =======================================  =====================================================================

MatchLSTM
#########

Model Documentation
*******************

Match LSTM model.

Examples:
    >>> model = MatchLSTM()
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.params['embedding_input_dim'] = 10000
    >>> model.params['embedding_output_dim'] = 100
    >>> model.params['embedding_trainable'] = True
    >>> model.params['fc_num_units'] = 200
    >>> model.params['lstm_num_units'] = 256
    >>> model.params['dropout_rate'] = 0.5
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================  ======================================================  ==========================================================================
  ..  Name                  Description                                                                                Default Value                                           Default Hyper-Space
====  ====================  =========================================================================================  ======================================================  ==========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.contrib.models.match_lstm.MatchLSTM'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                        adam
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  lstm_num_units        The hidden size in the LSTM layer.                                                         256                                                     quantitative uniform distribution in  [128, 384), with a step size of 32
   9  fc_num_units          The hidden size in the full connection layer.                                              200                                                     quantitative uniform distribution in  [100, 300), with a step size of 20
  10  dropout_rate          The dropout rate.                                                                          0.0                                                     quantitative uniform distribution in  [0.0, 0.9), with a step size of 0.01
====  ====================  =========================================================================================  ======================================================  ==========================================================================

ConvKNRM
########

Model Documentation
*******************

ConvKNRM model.

Examples:
    >>> model = ConvKNRM()
    >>> model.params['embedding_input_dim'] = 10000
    >>> model.params['embedding_output_dim'] = 300
    >>> model.params['embedding_trainable'] = True
    >>> model.params['filters'] = 128
    >>> model.params['conv_activation_func'] = 'tanh'
    >>> model.params['max_ngram'] = 3
    >>> model.params['use_crossmatch'] = True
    >>> model.params['kernel_num'] = 11
    >>> model.params['sigma'] = 0.1
    >>> model.params['exact_sigma'] = 0.001
    >>> model.guess_and_fill_missing_params(verbose=0)
    >>> model.build()

Model Hyper Parameters
**********************

====  ====================  =========================================================================================  ============================================  ===========================================================================
  ..  Name                  Description                                                                                Default Value                                 Default Hyper-Space
====  ====================  =========================================================================================  ============================================  ===========================================================================
   0  model_class           Model class. Used internally for save/load. Changing this may cause unexpected behaviors.  <class 'matchzoo.models.conv_knrm.ConvKNRM'>
   1  input_shapes          Dependent on the model and data. Should be set manually.
   2  task                  Decides model output shape, loss, and metrics.
   3  optimizer                                                                                                        adam
   4  with_embedding        A flag used help `auto` module. Shouldn't be changed.                                      True
   5  embedding_input_dim   Usually equals vocab size + 1. Should be set manually.
   6  embedding_output_dim  Should be set manually.
   7  embedding_trainable   `True` to enable embedding layer training, `False` to freeze embedding parameters.         True
   8  kernel_num            The number of RBF kernels.                                                                 11                                            quantitative uniform distribution in  [5, 20), with a step size of 1
   9  sigma                 The `sigma` defines the kernel width.                                                      0.1                                           quantitative uniform distribution in  [0.01, 0.2), with a step size of 0.01
  10  exact_sigma           The `exact_sigma` denotes the `sigma` for exact match.                                     0.001
  11  filters               The filter size in the convolution layer.                                                  128
  12  conv_activation_func  The activation function in the convolution layer.                                          relu
  13  max_ngram             The maximum length of n-grams for the convolution layer.                                   3
  14  use_crossmatch        Whether to match left n-grams and right n-grams of different lengths                       True
====  ====================  =========================================================================================  ============================================  ===========================================================================

