# -*- coding=utf-8 -*-
import os
import sys
import traceback


'''
def import_class(model_path, model_name):
    sys.path.append(model_path)
    comp = model_name.split('.')
    mod = __import__(comp[0])
    for co in comp[1:]:
        mod = getattr(mod, co)
    return mod

def import_class(model_path, model_name):
    sys.path.append(model_path)
    d = model_name.rfind('.')
    classname = model_name[d+1:len(model_name)]
    m = __import__(model_name[0:d], globals(), locals(), [classname])
    return getattr(m, classname)
'''

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
