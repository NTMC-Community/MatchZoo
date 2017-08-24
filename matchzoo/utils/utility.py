# -*- coding=utf-8 -*-
import os
import sys
import traceback

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
