"""DIIN model."""
import typing

import keras
import keras.backend as K
import tensorflow as tf

from matchzoo import preprocessors
from matchzoo.contrib.layers import DecayingDropoutLayer
from matchzoo.contrib.layers import EncodingLayer
from matchzoo.engine import hyper_spaces
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable


class DIIN(BaseModel):
    """
    DIIN model.

    Examples:
        >>> model = DIIN()
        >>> model.guess_and_fill_missing_params()
        >>> model.params['embedding_input_dim'] = 10000
        >>> model.params['embedding_output_dim'] = 300
        >>> model.params['embedding_trainable'] = True
        >>> model.params['optimizer'] = 'adam'
        >>> model.params['dropout_initial_keep_rate'] = 1.0
        >>> model.params['dropout_decay_interval'] = 10000
        >>> model.params['dropout_decay_rate'] = 0.977
        >>> model.params['char_embedding_input_dim'] = 100
        >>> model.params['char_embedding_output_dim'] = 8
        >>> model.params['char_conv_filters'] = 100
        >>> model.params['char_conv_kernel_size'] = 5
        >>> model.params['first_scale_down_ratio'] = 0.3
        >>> model.params['nb_dense_blocks'] = 3
        >>> model.params['layers_per_dense_block'] = 8
        >>> model.params['growth_rate'] = 20
        >>> model.params['transition_scale_down_ratio'] = 0.5
        >>> model.build()
    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params['optimizer'] = 'adam'
        params.add(Param(name='dropout_decay_interval', value=10000,
                         desc="The decay interval of decaying_dropout."))
        params.add(Param(name='char_embedding_input_dim', value=100,
                         desc="The input dimension of character embedding "
                              "layer."))
        params.add(Param(name='char_embedding_output_dim', value=2,
                         desc="The output dimension of character embedding "
                              "layer."))
        params.add(Param(name='char_conv_filters', value=8,
                         desc="The filter size of character convolution "
                              "layer."))
        params.add(Param(name='char_conv_kernel_size', value=2,
                         desc="The kernel size of character convolution "
                              "layer."))
        params.add(Param(name='first_scale_down_ratio', value=0.3,
                         desc="The channel scale down ratio of the "
                              "convolution layer before densenet."))
        params.add(Param(name='nb_dense_blocks', value=1,
                         desc="The number of blocks in densenet."))
        params.add(Param(name='layers_per_dense_block', value=2,
                         desc="The number of convolution layers in dense "
                              "block."))
        params.add(Param(name='growth_rate', value=2,
                         desc="The filter size of each convolution layer in "
                              "dense block."))
        params.add(Param(name='transition_scale_down_ratio', value=0.5,
                         desc="The channel scale down ratio of the "
                              "convolution layer in transition block."))
        params.add(Param(
            name='dropout_initial_keep_rate', value=1.0,
            hyper_space=hyper_spaces.quniform(
                low=0.8, high=1.0, q=0.02),
            desc="The initial keep rate of decaying_dropout."
        ))
        params.add(Param(
            name='dropout_decay_rate', value=0.97,
            hyper_space=hyper_spaces.quniform(
                low=0.90, high=0.99, q=0.01),
            desc="The decay rate of decaying_dropout."
        ))
        return params

    @classmethod
    def get_default_preprocessor(cls):
        """:return: Default preprocessor."""
        return preprocessors.DIINPreprocessor()

    def guess_and_fill_missing_params(self, verbose: int = 1):
        """
        Guess and fill missing parameters in :attr:'params'.

        Use this method to automatically fill-in hyper parameters.
        This involves some guessing so the parameter it fills could be
        wrong. For example, the default task is 'Ranking', and if we do not
        set it to 'Classification' manually for data packs prepared for
        classification, then the shape of the model output and the data will
        mismatch.

        :param verbose: Verbosity.
        """
        self._params.get('input_shapes').set_default([(32,),
                                                      (32,),
                                                      (32, 16),
                                                      (32, 16),
                                                      (32,),
                                                      (32,)], verbose)
        super().guess_and_fill_missing_params(verbose)

    def _make_inputs(self) -> list:
        text_left = keras.layers.Input(
            name='text_left',
            shape=self._params['input_shapes'][0]
        )
        text_right = keras.layers.Input(
            name='text_right',
            shape=self._params['input_shapes'][1]
        )
        char_left = keras.layers.Input(
            name='char_left',
            shape=self._params['input_shapes'][2]
        )
        char_right = keras.layers.Input(
            name='char_right',
            shape=self._params['input_shapes'][3]
        )
        match_left = keras.layers.Input(
            name='match_left',
            shape=self._params['input_shapes'][4]
        )
        match_right = keras.layers.Input(
            name='match_right',
            shape=self._params['input_shapes'][5]
        )
        return [text_left, text_right,
                char_left, char_right,
                match_left, match_right]

    def build(self):
        """Build model structure."""

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = word embedding size
        #   L = 'input_left' sequence length
        #   R = 'input_right' sequence length
        #   C = fixed word length

        inputs = self._make_inputs()
        # Left text and right text.
        # shape = [B, L]
        # shape = [B, R]
        text_left, text_right = inputs[0:2]
        # Left character and right character.
        # shape = [B, L, C]
        # shape = [B, R, C]
        char_left, char_right = inputs[2:4]
        # Left exact match and right exact match.
        # shape = [B, L]
        # shape = [B, R]
        match_left, match_right = inputs[4:6]

        # Embedding module
        left_embeddings = []
        right_embeddings = []

        # Word embedding feature
        word_embedding = self._make_embedding_layer()
        # shape = [B, L, D]
        left_word_embedding = word_embedding(text_left)
        # shape = [B, R, D]
        right_word_embedding = word_embedding(text_right)
        left_word_embedding = DecayingDropoutLayer(
            initial_keep_rate=self._params['dropout_initial_keep_rate'],
            decay_interval=self._params['dropout_decay_interval'],
            decay_rate=self._params['dropout_decay_rate']
        )(left_word_embedding)
        right_word_embedding = DecayingDropoutLayer(
            initial_keep_rate=self._params['dropout_initial_keep_rate'],
            decay_interval=self._params['dropout_decay_interval'],
            decay_rate=self._params['dropout_decay_rate']
        )(right_word_embedding)
        left_embeddings.append(left_word_embedding)
        right_embeddings.append(right_word_embedding)

        # Exact match feature
        # shape = [B, L, 1]
        left_exact_match = keras.layers.Reshape(
            target_shape=(K.int_shape(match_left)[1], 1,)
        )(match_left)
        # shape = [B, R, 1]
        right_exact_match = keras.layers.Reshape(
            target_shape=(K.int_shape(match_left)[1], 1,)
        )(match_right)
        left_embeddings.append(left_exact_match)
        right_embeddings.append(right_exact_match)

        # Char embedding feature
        char_embedding = self._make_char_embedding_layer()
        char_embedding.build(
            input_shape=(None, None, K.int_shape(char_left)[-1]))
        left_char_embedding = char_embedding(char_left)
        right_char_embedding = char_embedding(char_right)
        left_embeddings.append(left_char_embedding)
        right_embeddings.append(right_char_embedding)

        # Concatenate
        left_embedding = keras.layers.Concatenate()(left_embeddings)
        right_embedding = keras.layers.Concatenate()(right_embeddings)
        d = K.int_shape(left_embedding)[-1]

        # Encoding module
        left_encoding = EncodingLayer(
            initial_keep_rate=self._params['dropout_initial_keep_rate'],
            decay_interval=self._params['dropout_decay_interval'],
            decay_rate=self._params['dropout_decay_rate']
        )(left_embedding)
        right_encoding = EncodingLayer(
            initial_keep_rate=self._params['dropout_initial_keep_rate'],
            decay_interval=self._params['dropout_decay_interval'],
            decay_rate=self._params['dropout_decay_rate']
        )(right_embedding)

        # Interaction module
        interaction = keras.layers.Lambda(self._make_interaction)(
            [left_encoding, right_encoding])

        # Feature extraction module
        feature_extractor_input = keras.layers.Conv2D(
            filters=int(d * self._params['first_scale_down_ratio']),
            kernel_size=(1, 1),
            activation=None)(interaction)
        feature_extractor = self._create_densenet()
        features = feature_extractor(feature_extractor_input)

        # Output module
        features = DecayingDropoutLayer(
            initial_keep_rate=self._params['dropout_initial_keep_rate'],
            decay_interval=self._params['dropout_decay_interval'],
            decay_rate=self._params['dropout_decay_rate'])(features)
        out = self._make_output_layer()(features)

        self._backend = keras.Model(inputs=inputs, outputs=out)

    def _make_char_embedding_layer(self) -> keras.layers.Layer:
        """
        Apply embedding, conv and maxpooling operation over time dimension
        for each token to obtain a vector.

        :return: Wrapper Keras 'Layer' as character embedding feature
            extractor.
        """

        return keras.layers.TimeDistributed(keras.Sequential([
            keras.layers.Embedding(
                input_dim=self._params['char_embedding_input_dim'],
                output_dim=self._params['char_embedding_output_dim'],
                input_length=self._params['input_shapes'][2][-1]),
            keras.layers.Conv1D(
                filters=self._params['char_conv_filters'],
                kernel_size=self._params['char_conv_kernel_size']),
            keras.layers.GlobalMaxPooling1D()]))

    def _make_interaction(self, inputs_) -> typing.Any:
        left_encoding = inputs_[0]
        right_encoding = inputs_[1]

        left_encoding = tf.expand_dims(left_encoding, axis=2)
        right_encoding = tf.expand_dims(right_encoding, axis=1)

        interaction = left_encoding * right_encoding
        return interaction

    def _create_densenet(self) -> typing.Callable:
        """
        DenseNet is consisted of 'nb_dense_blocks' sets of Dense block
        and Transition block pair.

        :return: Wrapper Keras 'Layer' as DenseNet, tensor in tensor out.
        """
        def _wrapper(x):
            for _ in range(self._params['nb_dense_blocks']):
                # Dense block
                # Apply 'layers_per_dense_block' convolution layers.
                for _ in range(self._params['layers_per_dense_block']):
                    out_conv = keras.layers.Conv2D(
                        filters=self._params['growth_rate'],
                        kernel_size=(3, 3),
                        padding='same',
                        activation='relu')(x)
                    x = keras.layers.Concatenate(axis=-1)([x, out_conv])

                # Transition block
                # Apply a convolution layer and a maxpooling layer.
                scale_down_ratio = self._params['transition_scale_down_ratio']
                nb_filter = int(K.int_shape(x)[-1] * scale_down_ratio)
                x = keras.layers.Conv2D(
                    filters=nb_filter,
                    kernel_size=(1, 1),
                    padding='same',
                    activation=None)(x)
                x = keras.layers.MaxPool2D(strides=(2, 2))(x)

            out_densenet = keras.layers.Flatten()(x)
            return out_densenet

        return _wrapper
