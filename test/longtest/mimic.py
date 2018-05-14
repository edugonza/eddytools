from test import test_disc
from eddytools import extraction as ex


def test_extraction(connection_params, openslex_file_path, classes=None):
    ex.extract_to_mm(openslex_file_path, **connection_params,
                     classes=classes, overwrite=True)


def test_discovery(connection_params, dump_dir):
    test_disc(connection_params, dump_dir=dump_dir)  # , classes_for_fk=['public.admissions'])


def test_extraction_custom_datamodel(connection_params, openslex_file_path, classes=None):
    meta = None
    ex.extract_to_mm(openslex_file_path, **connection_params, classes=classes, overwrite=True, metadata=meta)


if __name__ == '__main__':

    connection_params = {
        'dialect': 'postgresql',
        'username': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5555',
        'database': 'mimic',
        'schema': 'public',
    }

    openslex_file_path = 'data/output/sample-mimic.slexmm'

    classes_extract = ['patients', 'admissions', 'microbiologyevents',
               'd_items']  # use this to specify a subset of the classes

    test_extraction(connection_params, openslex_file_path, classes_extract)

    dump_dir = 'data/output/dump-mimic-sample-04'

    test_discovery(connection_params, dump_dir)

    test_extraction_custom_datamodel(connection_params, openslex_file_path, classes_extract)
