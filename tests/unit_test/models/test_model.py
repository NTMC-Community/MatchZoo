"""Test basic class."""

import os
import unittest
from matchzoo.models.model import BasicModel


class TestBasicModel(unittest.TestCase):
    """Test basic model."""

    def test_base_model(self):
        """Test set default function."""
        config = {'fake_key': 'fake_value'}
        base_model = BasicModel(config)
        base_model.set_default('fake_key_2', 'fake_value_2')
        self.assertEqual(
            base_model.config['fake_key_2'],
            'fake_value_2')
