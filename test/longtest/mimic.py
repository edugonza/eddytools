# from test import test_disc
from eddytools import extraction as ex


def test_extraction(connection_params, openslex_file_path, cache_dir, schemas=None, classes=None):
    ex.extract_to_mm(openslex_file_path, connection_params, cache_dir=cache_dir,
                     classes=classes, overwrite=True, schemas=schemas)


# def test_discovery(connection_params, dump_dir):
#     test_disc(connection_params, dump_dir=dump_dir)  # , classes_for_fk=['public.admissions'])


# def test_extraction_custom_datamodel(connection_params, openslex_file_path, cache_dir, classes=None):
#     meta = None
#     ex.extract_to_mm(openslex_file_path, **connection_params, cache_dir=cache_dir,
#                      classes=classes, overwrite=True, metadata=meta)


if __name__ == '__main__':

    connection_params = {
        'dialect': 'postgresql',
        'username': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5555',
        'database': 'mimic',
    }

    schemas = ['public']

    dir = 'output/mimic'

    openslex_file_path = '{}/sample-mimic.slexmm'.format(dir)

    # classes_extract = ['patients', 'admissions', 'microbiologyevents',
    #            'd_items']  # use this to specify a subset of the classes

    test_extraction(connection_params, openslex_file_path, dir, schemas)

    # dump_dir = 'data/output/dump-mimic-sample-04'

    # test_discovery(connection_params, dump_dir)

    # test_extraction_custom_datamodel(connection_params, openslex_file_path, dir, classes_extract)
