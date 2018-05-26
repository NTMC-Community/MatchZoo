"""Contains the base Model class, from which all models inherit.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import abc


class BaseModel(object):
    """Abstract base model class.

    # Properties:
        name: str, name of the model.
        objective: str, model objective, ranking or classification.
        trainable: boolean, indicates whether the model is allowed to train. 
        fixed_hyper_parameters: dict, fixed hyper parameters with values.
        default_hyper_parameters: dict, universal hyper parameters.
        model_specific_hyper_parameters: dict, hyper parameters w.r.t models.
        user_given_parameters: dict, hyper parameters given by users.
        num_default_hidden_layers: int, default number hidden layer in paper.
        num_custom_hidden_layers: int, custom number hidden layer decided by user.

    # Methods:
        train(text_1, text_2, labels)

    # Internal Methods:
        _aggregate_hyper_parameters()
        _build()
        _compile(model)
    """

    def __init__(self, **kwargs):
        """Initialization"""
        self._name = 'BaseModel'
        self._objective = None
        self._trainable = True
        #  This list serve as a "BLACK LIST", not allowed to changed.
        self._list_fixed_hyper_parameters = []
        self._fixed_hyper_parameters = {}
        # Universal model parameters, can be overwrite if not fixed.
        self._default_hyper_parameters = {
            # TODO ADD DEFAULT PARAMETERS
        }
        # Model specific hyper parameters, can be overwritten by
        # _user_given_parameters if not fixed.
        self._model_specific_hyper_parameters = {}
        # User given hyper parameters.
        # THE Intersection between default_hyper_parameters and
        # _model_specific_hyper_parameters will be overrite by
        # _user_given_parameters if it's not fixed
        self._user_given_parameters = {}
        for key, value in kwargs.iteritems():
            if key not in self._list_fixed_hyper_parameters:
                self._user_given_parameters[key] = value
        # Number of default hidden layers and extra hidden layers.
        # Extra hidden layers is used when user customize model.
        # E.g. DSSM is a 3 layer network, user can custom a 5 layer
        #   DSSM with _num_custom_hidden_layers.
        # Num_default_hidden_layers should < num_custom_hidden_layers.
        self._num_default_hidden_layers = None
        self._num_custom_hidden_layers = None

    @property
    def name(self):
        """Get model name."""
        return self._name

    @property
    def objective(self):
        """Model objective, classification, ranking or both."""
        return self._objective

    @objective.setter
    def objective(self, value):
        """Set model objective."""
        if value not in ['ranking', 'classification']:
            raise ValueError('{} is not a valid model objective'.format(
                value))
        self._objective = value

    @property
    def trainable(self):
        """Indicate allow to train or not."""
        return self._trainable

    @property
    def fixed_hyper_parameters(self):
        """Get fixed hyper parameters."""
        return self._fixed_hyper_parameters

    @fixed_hyper_parameters.setter
    def fixed_hyper_parameters(self, **kwargs):
        """Set fixed hyper parameters"""
        for key, value in kwargs.iteritems():
            self._fixed_hyper_parameters[key] = value
            self._list_fixed_hyper_parameters.append(key)

    @property
    def default_hyper_parameters(self):
        """Universal parameters that can be use across varies models.
        """
        return self._default_parameters

    @default_hyper_parameters.setter
    def default_hyper_parameters(self, **kwargs):
        """Set default hyper parameters."""
        allowed_default_parameters = self._default_hyper_parameters.keys()
        for key, value in kwargs.iteritems():
            if key not in allowed_default_parameters:
                raise ValueError(
                    '{} not in allowed default parameters: {}.'.format(
                        key,
                        allowed_default_parameters))
            self._default_hyper_parameters[key] = value

    @property
    def model_specific_hyper_parameters(self):
        """Get model specific hyper parameters."""
        return self._model_specific_hyper_parameters

    @model_specific_hyper_parameters.setter
    def model_specific_hyper_parameters(self, **kwargs):
        """Set model specific hyper parameters."""
        allowed_model_parameters = self._model_specific_hyper_parameters.keys()
        for key, value in kwargs.iteritems():
            if key not in allowed_model_parameters:
                raise ValueError(
                    '{} not in allowed model specific parameters: {}'.format(
                        key,
                        allowed_model_parameters))
            self._model_specific_hyper_parameters[key] = value

    @property
    def user_given_parameters(self):
        """Get user given hyper parameters."""
        return self._user_given_parameters

    def _aggregate_hyper_parameters(self):
        """This method is used to merge all the parameters.

        This function is used internaly.`

        # Returns:
            conf: final model  configuration.
        """
        conf = self._default_hyper_parameters.copy()
        # Universal config will be overwrite by model_specific.
        # Then merge.
        conf.update(self._model_specific_hyper_parameters)
        # Merged universal config and model config will be.
        # overriten by user_given_parameters.
        conf.update(self._user_given_parameters)
        # Filter fixed-parameters from configuration.
        [conf.pop(key, None) for key in self._list_fixed_hyper_parameters]
        # Merge default fixed parameters.
        conf.update(self._fixed_hyper_parameters)
        return conf

    @property
    def num_default_hidden_layers(self):
        """Get number of default hiddden layers."""
        return self._num_default_hidden_layers

    @num_default_hidden_layers.setter
    def num_default_hidden_layers(self, value):
        """Set number of default hidden layers."""
        if self._num_custom_hidden_layers:
            if self._num_default_hidden_layers > self._num_custom_hidden_layers:
                raise ValueError(
                    'Number of default hidden layers should smaller than extended \
                     hidden layers, get ({},{})'.format(
                        self._num_default_hidden_layers,
                        self._num_custom_hidden_layers))

        self._num_default_hidden_layers = value

    @property
    def num_custom_hidden_layers(self):
        """Get the number of custom hidden layers."""
        return self._num_custom_hidden_layers

    @num_custom_hidden_layers.setter
    def num_custom_hidden_layers(self, value):
        """Set the number of custom hideen layers."""
        if not self._num_default_hidden_layers:
            raise TypeError(
                'Number of default hidden layer expected, None found.')
        if value < self._num_default_hidden_layers:
            raise ValueError(
                'Number of default hidden layer greater than \
        		 custom hidden layers ({},{})'.format(
                    self._num_default_hidden_layers,
                    value))
        self._num_custom_hidden_layers = value

    @abc.abstractmethod
    def _build(self):
        """Build model, each sub class need to impelemnt this method.

        This function is used internally with `self._build()`.
        """
        return

    @abc.abstractmethod
    def _compile(self, model):
        """Compile model, each  sub class need to implement this method.

        This function is used internally with `self._compile()`.

        # Arguments:
            model: keras Model instance.
        """
        return

    @abc.abstractmethod
    def train(self, text_1, text_2, labels):
        """Train MatchZoo model, each sub class need to implement this method.

        # Arguments:
            text_1: list of text to be trained, left node.
            text_2: list of text to be trained, right node.
            labels: ground truth.
        """
        return
