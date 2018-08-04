import os
import pytest

from matchzoo import datapack
from matchzoo import generators
from matchzoo import preprocessor
from matchzoo import models

def prepare_data():
	"""Prepare train & test data."""
	train = []
	test = []
	path = os.path.dirname(__file__)
	with open(os.path.join(path, 'train.txt')) as f:
		train = [tuple(map(str, i.split('\t'))) for i in f]
	with open(os.path.join(path, 'test.txt')) as f:
		test = [tuple(map(str, i.split('\t'))) for i in f]
	return train, test


def inte_test_dssm():
	"""Test DSSM model."""
	# load data.
	train, test = prepare_data()
	# do pre-processing.
	dssm_preprocessor = preprocessor.DSSMPreprocessor()
	processed_train = dssm_preprocessor.fit_transform(train, stage='train')
	# generator.
	generator = generators.PointGenerator(processed_train)
	# TODO GENERATOR
	


inte_test_dssm()
