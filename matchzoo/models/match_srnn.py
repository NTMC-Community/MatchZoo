"""An implementation of Match-SRNN Model."""
import typing

import keras
import keras.backend as K

import matchzoo
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.param import Param
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine import hyper_spaces


class MatchSRNN(BaseModel):
    """
    Match-SRNN Model.

    Examples:
        >>> model = MatchSRNN()
        >>> model.params['channels'] = 4
        >>> model.params['units'] = 10
        >>> model.params['dropout_rate'] = 0.0
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
        # shape = [B, L]
        # shape = [B, R]
        query, doc = self._make_inputs()

        embedding = self._make_embedding_layer()
        # Process left and right input.
        # shape = [B, L, D]
        embed_query = embedding(query)
        # shape = [B, R, D]
        embed_doc = embedding(doc)

        # Get matching tensor
        # shape = [B, C, L, R]
        matching_tensor_layer = matchzoo.layers.MatchingTensorLayer(
            channels=self._params['channels'])
        matching_tensor = matching_tensor_layer([embed_query, embed_doc])
        # shape = [B, L, R, C]
        matching_tensor = keras.layers.Permute((2, 3, 1))(matching_tensor)

        spatial_gru = matchzoo.layers.SpatialGRU(
            units=self._params['units'])

        h_ij = spatial_gru(matching_tensor)

        x = keras.layers.Dropout(
            rate=self._params['dropout_rate'])(h_ij)

        x_out = self._make_output_layer()(x)

        self._backend = keras.Model(inputs=[query, doc], outputs=x_out)
