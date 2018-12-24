"""Match LSTM model."""
import keras
import keras.backend as K

from matchzoo import engine


class MatchLSTM(engine.BaseModel):
    """
    Match LSTM model.

    Examples:
        >>> model = MatchLSTM()
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()
        >>> model.params['fc_hidden_size'] = 200
        >>> model.params['rnn_hidden_size'] = 256

    """

    @classmethod
    def get_default_params(cls):
        """Get default parameters."""
        params = super().get_default_params(with_embedding=True)
        params.add(engine.Param(
            'rnn_hidden_size', 256,
            hyper_space=engine.hyper_spaces.quniform(low=128, high=384, q=32)
        ))
        params.add(engine.Param(
            'fc_hidden_size', 200,
            hyper_space=engine.hyper_spaces.quniform(
                low=100, high=300, q=20)
        ))
        return params

    def build(self):
        """Build model."""
        query, doc = self._make_inputs()
        query_len = query.shape[1]
        doc_len = doc.shape[1]
        embedding = self._make_embedding_layer()
        query_embed = embedding(query)
        doc_embed = embedding(doc)

        lstm_query = keras.layers.LSTM(self._params['rnn_hidden_size'],
                                       return_sequences=True,
                                       name='lstm_query')
        lstm_doc = keras.layers.LSTM(self._params['rnn_hidden_size'],
                                     return_sequences=True,
                                     name='lstm_doc')
        doc_encoded = lstm_doc(doc_embed)
        query_encoded = lstm_query(query_embed)

        def attention(tensors):
            """Attention layer."""
            query, doc = tensors
            tensor_left = K.expand_dims(query, axis=2)
            tensor_right = K.expand_dims(doc, axis=1)
            tensor_left = K.repeat_elements(tensor_left, doc_len, 2)
            tensor_right = K.repeat_elements(tensor_right, query_len, 1)
            tensor_merged = K.concatenate([tensor_left, tensor_right], axis=-1)
            middle_output = keras.layers.Dense(self._params['fc_hidden_size'],
                                               activation='tanh')(
                tensor_merged)
            attn_scores = keras.layers.Dense(1)(middle_output)
            attn_scores = K.squeeze(attn_scores, axis=3)
            exp_attn_scores = K.exp(
                attn_scores - K.max(attn_scores, axis=-1, keepdims=True))
            exp_sum = K.sum(exp_attn_scores, axis=-1, keepdims=True)
            attention_weights = exp_attn_scores / exp_sum
            return K.batch_dot(attention_weights, doc)

        attn_layer = keras.layers.Lambda(attention)
        query_attn_vec = attn_layer([query_encoded, doc_encoded])
        concat = keras.layers.Concatenate(axis=2)(
            [query_attn_vec, doc_encoded])
        lstm_merge = keras.layers.LSTM(self._params['rnn_hidden_size'] * 2,
                                       return_sequences=True,
                                       name='lstm_merge')
        merged = lstm_merge(concat)
        phi = keras.layers.Dense(self._params['fc_hidden_size'],
                                 activation='tanh')(merged)
        out = self._make_output_layer()(phi)
        self._backend = keras.Model(inputs=[query, doc], outputs=[out])
