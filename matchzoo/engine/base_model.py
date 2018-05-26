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
        objective: str, model objective, ranking or classification
        trainable: boolean, indicates whether the model is allowed to train. 
        fixed_hyper_parameters: list, fixed hyper parameters.
        default_hyper_parameters: dict, default hyper parameters described in paper.
        to be discussed


    # Methods:
        train(text_1, text_2, labels)

    # Internal Methods:
        _aggregate_parameters(default, model_specific, user_given)
        _build()
        _compile(model)
    """

    def __init__(self, **kwargs):
        """Initialization"""
        self._name = 'BaseModel'
        self._objective = None
        self._trainable = True
        self._fixed_hyper_parameters = [
            #  TODO ADD fixed hyper parameters.
            #  THESE PARAMETERS SHOULD NOT BE TUNABLE.
            #  IF USER PASS, RAISE EXCEPTION
            #  THIS VARIABLE IS USED TO CHECK self._default_parameters
            #  and self._custom_hyper_parameters
            #  THIS LIST SERVE AS A "BLACK LIST",
            #  THE REST CAN BE TREAT AS TUNABLE PARAMETERS
        ]
        # DEFINE UNIVERSAL MODEL DEFAULT PARAMETERS
        self._default_hyper_parameters = {
            # TODO ADD DEFAULT PARAMETERS
        }
        # MODEL SPECIFIC HYPER PARAMETERS
        self._model_specific_hyper_parameters = {}  # ADD PARAMETER PER MODEL
        # USER GIVEN HYPER PARAMETERS
        # THE Intersection between default_hyper_parameters and
        #  _model_specific_hyper_parameters
        # should be overrite by _user_given_parameters if it's not fixed
        self._user_given_parameters = {}
        for key, value in kwargs.iteritems():
            if key not in self._fixed_hyper_parameters:
                self._user_given_parameters[key] = value
        # DEFINE NUMBER OF HIDDEN LAYERS, IF NONE,
        # USE DEFAULT AS PAPER DESCRIBED, ELSE CREATE EXTRA HIDDEN LAYER
        # USUALLY _num_default_hidden_layers > _num_extra_hidden_layers
        # THESE TWO VARIABLES IS DESIGNED TO ALLOW USER TO ADD MODE LAYERS
        self._num_default_hidden_layers = None
        self._num_extra_hidden_layers = None

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
    def default_hyper_parameters(self):
        """Universal parameters that can be use across varies models.

        Default hyper parameters can be overwritten
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
        """This method is used to merge all the parameters

        default -> model_specific -> user_given
        low_level hyper parameters can be overwritten by high level
        parameters if the parameter is not FIXED.
        """
        pass

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


# AN EXAMPLE USAGE
# base_model = BaseModel(train_test_split=1,
#                         learning_rate=0.1)
# we can provide list of allowed parameters in documentation.
# train_test_split will be user_given_parameters
# and overwrite default & model specific
