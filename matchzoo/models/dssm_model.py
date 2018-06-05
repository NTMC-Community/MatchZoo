"""DSSM, Deep Structured Semantic Model."""

from matchzoo import engine
from keras import optimizers


class DssmModel(engine.BaseModel):
	"""
	Deep structured semantic model.

	Extends:
		engine.BaseModel
	"""

    @classmethod
    def get_default_params(cls) -> engine.ModelParams:
    	""":return: model default parameters."""
    	params = engine.ModelParams()
    	params['name'] = 'DSSM'
    	params['w_initializer'] = 'glorot_normal'
    	params['b_initializer'] = 'zeros'
        # TODO GET TRI-LETTER DIMENSIONALITY FROM FIT-TRANSFORM
    	params['input_shapes'] = [(1,), (1,)]
    	params['dim_fan_out'] = 128
    	params['dim_hidden'] = 300
    	params['activation_hidden'] = 'tanh'
    	params['activation_prediction'] = 'softmax'
    	params['lr'] = 0.01
    	params['decay'] = 1e-6
    	params['momentum'] = 0.9
    	params['nesterov'] = True
    	params['batch_size'] = 1024
    	params['num_epochs'] = 20
    	params['train_test_split'] = 0.2
    	params['verbose'] = 1
    	params['shuffle'] = True
    	params['optimizer'] = optimizers.SGD
    	params['metrics'] = ['accuracy']
    	params['reg_rate'] = 0
    	params['dropout_rate'] = 0
    	params['sim'] = 'cosine'
    	return params

    def build(self):
    	"""
    	Build model structure.
    	
    	DSSM use pair-wise arthitecture.
    	"""
    	x_in = [self._shared(), self._shared()]
        # Dot product with cosine similarity.
        x = Dot(axis=1,
        	    normalize=True)(x_in)
        x_out = Activation(activation=self._params['activation_prediction'])(x)
        self._backend = keras.models.Model(inputs=x_in, outputs=x_out)

    def _shared(self):
        """
        
        Common architecture share to adopt pair-wise input.

        Word-hashing layer, hidden layer * 2,  out layer.
        
        :returns:  128 dimension vector representation.
        """
        x_in = keras.layers.Input(
            self.params['input_shapes'][0])
        x = Dense(self._params['dim_hidden'],
                  input_shape=self._params['input_shapes'][0],
                  kernel_initializer=self._params['w_initializer'],
                  bias_initializer=self._params['b_initializer'])(x_in)
        x = Dense(self._params['dim_hidden'],
                  input_shape=(self._params['dim_hidden'],),
                  activation=self._params['activation_hidden'])
        x = Dense(self._params['dim_hidden'],
                  input_shape=(self._params['dim_hidden'],),
                  activation=self._params['activation_hidden'])
        x_out = Dense(self._params['dim_fan_out'],
                  input_shape=(self._params['dim_hidden'],),
                  activation=self._params['activation_hidden'])
        return x_out
