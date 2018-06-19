# -*- coding=utf-8 -*-
import os
import sys
import traceback
import psutil


def show_layer_info(layer_name, layer_out):
    print('[layer]: %s\t[shape]: %s \n%s' % (layer_name,str(layer_out.get_shape().as_list()), show_memory_use()))


def show_memory_use():
    used_memory_percent = psutil.virtual_memory().percent
    strinfo = '{}% memory has been used'.format(used_memory_percent)
    return strinfo


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                (class_str,
                    traceback.format_exception(*sys.exc_info())))


def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)


def import_module(import_str):
    __import__(import_str)
    return sys.modules[import_str]
