# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
from utils.rank_io import *
from .rank_data_generator import PairGenerator

class DRMM_PairGenerator(PairGenerator):
    def __init__(self, hist_size, data1, data2, config):
        self.hist_size = hist_size
        super(DRMM_PairGenerator, self).__init__(data1, data2, config)
