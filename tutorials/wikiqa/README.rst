**********************
WikiQA Best Parameters
**********************

DRMMTKS
#######

====  ===========================  =========================================
  ..  Name                         Value
====  ===========================  =========================================
   0  model_class                  <class 'matchzoo.models.drmmtks.DRMMTKS'>
   1  input_shapes                 [(10,), (100,)]
   2  task                         Ranking Task
   3  optimizer                    adadelta
   4  with_embedding               True
   5  embedding_input_dim          16674
   6  embedding_output_dim         100
   7  embedding_trainable          True
   8  with_multi_layer_perceptron  True
   9  mlp_num_units                5
  10  mlp_num_layers               1
  11  mlp_num_fan_out              1
  12  mlp_activation_func          relu
  13  mask_value                   -1
  14  top_k                        20
====  ===========================  =========================================

MatchPyramid
############

====  ====================  ====================================================
  ..  Name                  Value
====  ====================  ====================================================
   0  model_class           <class 'matchzoo.models.match_pyramid.MatchPyramid'>
   1  input_shapes          [(10,), (40,)]
   2  task                  Ranking Task
   3  optimizer             adam
   4  with_embedding        True
   5  embedding_input_dim   16546
   6  embedding_output_dim  100
   7  embedding_trainable   True
   8  num_blocks            2
   9  kernel_count          [16, 32]
  10  kernel_size           [[3, 3], [3, 3]]
  11  activation            relu
  12  dpool_size            [3, 10]
  13  padding               same
  14  dropout_rate          0.1
====  ====================  ====================================================

ArcII
#####

====  ====================  =====================================
  ..  Name                  Value
====  ====================  =====================================
   0  model_class           <class 'matchzoo.models.arcii.ArcII'>
   1  input_shapes          [(10,), (100,)]
   2  task                  Ranking Task
   3  optimizer             adam
   4  with_embedding        True
   5  embedding_input_dim   16674
   6  embedding_output_dim  100
   7  embedding_trainable   True
   8  num_blocks            2
   9  kernel_1d_count       32
  10  kernel_1d_size        3
  11  kernel_2d_count       [64, 64]
  12  kernel_2d_size        [3, 3]
  13  activation            relu
  14  pool_2d_size          [[3, 3], [3, 3]]
  15  padding               same
  16  dropout_rate          0.0
====  ====================  =====================================

DUET
####

====  ====================  ===================================
  ..  Name                  Value
====  ====================  ===================================
   0  model_class           <class 'matchzoo.models.duet.DUET'>
   1  input_shapes          [(10,), (100,)]
   2  task                  Ranking Task
   3  optimizer             adagrad
   4  with_embedding        True
   5  embedding_input_dim   16674
   6  embedding_output_dim  100
   7  embedding_trainable   True
   8  lm_filters            32
   9  lm_hidden_sizes       [32]
  10  dm_filters            32
  11  dm_kernel_size        3
  12  dm_q_hidden_size      32
  13  dm_d_mpool            4
  14  dm_hidden_sizes       [32]
  15  padding               same
  16  activation_func       relu
  17  dropout_rate          0.5
====  ====================  ===================================

