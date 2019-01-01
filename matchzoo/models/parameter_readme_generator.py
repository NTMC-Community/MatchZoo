"""matchzoo/models/README.md generater."""

from pathlib import Path

import markdown_generator

import matchzoo


def _get_table_skeleton():
    table = markdown_generator.Table()
    table.add_column('name', markdown_generator.Alignment.RIGHT)
    table.add_column('description')
    table.add_column('default value')
    table.add_column('default hyper space')
    return table


def _make_table(params):
    table = _get_table_skeleton()
    for param in params:
        name = param.name
        desc = param.desc or ''
        val = param.value or ''
        if param.hyper_space is None:
            hyper = ''
        else:
            hyper = str(param.hyper_space)
        table.append(name, desc, val, hyper)
    return str(table)


if __name__ == '__main__':
    full = '# MatchZoo Model Hyper-parameters Reference \n\n'
    full += '## Shared Parameters \n\n'
    shared = matchzoo.engine.BaseModel.get_default_params()
    full += _make_table(shared)

    for m in matchzoo.models.list_available():
        m = m()
        full += '## ' + str(m.__class__.__name__) + '\n\n'
        full += _make_table([p for p in m.get_default_params()
                             if p.name not in shared.keys()])

    file_path = Path(__file__).parent.joinpath('README.md')
    with open(file_path, 'w', encoding='utf-8') as out_file:
        out_file.write(full)
