"""matchzoo/models/README.md generater."""

from pathlib import Path

import tabulate
import inspect
import pandas as pd

import matchzoo


def _generate():
    full = _make_title()
    for model_class in matchzoo.models.list_available():
        full += _make_model_class_subtitle(model_class)
        full += _make_doc_section_subsubtitle()
        full += _make_model_doc(model_class)
        model = model_class()
        full += _make_params_section_subsubtitle()
        full += _make_model_params_table(model)
    _write_to_files(full)


def _make_title():
    title = 'MatchZoo Model Reference'
    line = '*' * len(title)
    return line + '\n' + title + '\n' + line + '\n\n'


def _make_model_class_subtitle(model_class):
    subtitle = model_class.__name__
    line = '#' * len(subtitle)
    return subtitle + '\n' + line + '\n\n'


def _make_doc_section_subsubtitle():
    subsubtitle = 'Model Documentation'
    line = '*' * len(subsubtitle)
    return subsubtitle + '\n' + line + '\n\n'


def _make_params_section_subsubtitle():
    subsubtitle = 'Model Hyper Parameters'
    line = '*' * len(subsubtitle)
    return subsubtitle + '\n' + line + '\n\n'


def _make_model_doc(model_class):
    return inspect.getdoc(model_class) + '\n\n'


def _make_model_params_table(model):
    params = model.get_default_params()
    df = pd.DataFrame(data={
        'Name': [p.name for p in params],
        'Description': [p.desc for p in params],
        'Default Value': [p.value for p in params],
        'Default Hyper-Space': [p.hyper_space for p in params]
    }, columns=['Name', 'Description', 'Default Value', 'Default Hyper-Space'])
    return tabulate.tabulate(df, tablefmt='rst', headers='keys') + '\n\n'


def _write_to_files(full):
    readme_file_path = Path(__file__).parent.joinpath('README.rst')
    doc_file_path = Path(__file__).parent.parent.parent. \
        joinpath('docs').joinpath('source').joinpath('model_reference.rst')
    for file_path in readme_file_path, doc_file_path:
        with open(file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(full)


if __name__ == '__main__':
    _generate()
