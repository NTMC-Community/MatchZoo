"""An implementation of Match-SRNN Model."""

import keras

from matchzoo.contrib.layers import MatchingTensorLayer
from matchzoo.contrib.layers import SpatialGRU
from matchzoo.engine import hyper_spaces
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable


class MatchSRNN(BaseModel):
    """
    Match-SRNN Model.

    Examples:
        >>> model = MatchSRNN()
        >>> model.params['channels'] = 4
        >>> model.params['units'] = 10
        >>> model.params['dropout_rate'] = 0.0
        >>> model.params['direction'] = 'lt'
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    @classmethod
    def get_default_params(cls) -> ParamTable:
        """:return: model default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(Param(name='channels', value=4,
                         desc="Number of word interaction tensor channels"))
        params.add(Param(name='units', value=10,
                         desc="Number of SpatialGRU units"))
        params.add(Param(name='direction', value='lt',
                         desc="Direction of SpatialGRU scanning"))
        params.add(Param(
            name='dropout_rate', value=0.0,
            hyper_space=hyper_spaces.quniform(low=0.0, high=0.8,
                                              q=0.01),
            desc="The dropout rate."
        ))
        return params

    def build(self):
        """
        Build model structure.

        Match-SRNN: Modeling the Recursive Matching Structure
            with Spatial RNN
        """

        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   C = number of channels

        # Left input and right input.
        # query = [B, L]
        # doc = [B, R]
        query, doc = self._make_inputs()

        # Process left and right input.
        # embed_query = [B, L, D]
        # embed_doc = [B, R, D]
        embedding = self._make_embedding_layer()
        embed_query = embedding(query)
        embed_doc = embedding(doc)

        # Get matching tensor
        # matching_tensor = [B, C, L, R]
        matching_tensor_layer = MatchingTensorLayer(
            channels=self._params['channels'])
        matching_tensor = matching_tensor_layer([embed_query, embed_doc])

        # Apply spatial GRU to the word level interaction tensor
        # h_ij = [B, U]
        spatial_gru = SpatialGRU(
            units=self._params['units'],
            direction=self._params['direction'])
        h_ij = spatial_gru(matching_tensor)

        # Apply Dropout
        x = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(h_ij)

        # Make output layer
        x_out = self._make_output_layer()(x)

        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
