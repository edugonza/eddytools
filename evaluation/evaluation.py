from eddytools import schema as es


def test_disc_mimic(resume=False, sampling=0, max_fields_key=2, dump_dir=None):

    connection_params = {
        'dialect': 'postgresql',
        'username': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5558',
        'database': 'mimic',
    }

    schemas = ['public']

    return es.full_discovery(connection_params, schemas=schemas, dump_dir=dump_dir,
                             max_fields_key=max_fields_key, resume=resume, sampling=sampling)


def test_disc_mssql(resume=False, sampling=0, max_fields_key=2, dump_dir=None):

    connection_params = {
        'dialect': 'mssql+pymssql',
        'username': 'SA',
        'password': 'msSQL2017!',
        'host': 'localhost',
        'port': '1402',
        'database': 'AdventureWorks2017',
        'trusted_conn': False,
        'timeout': 600,
    }

    return es.full_discovery(connection_params, schemas=None, dump_dir=dump_dir,
                             max_fields_key=max_fields_key, resume=resume, sampling=sampling)


def test_disc_ds2(resume=False, sampling=0, max_fields_key=2, dump_dir=None):

    connection_params = {
        'dialect': 'postgresql',
        'username': 'ds2',
        'password': 'ds2',
        'host': 'localhost',
        'port': '5556',
        'database': 'ds2',
    }

    schemas = ['public']

    return es.full_discovery(connection_params, schemas=schemas, dump_dir=dump_dir,
                             max_fields_key=max_fields_key, resume=resume, sampling=sampling)


if __name__ == '__main__':

    # test_disc_ds2(resume=False, sampling=0, max_fields_key=1, dump_dir='output/ds2/dumps-ns-1/')
    # test_disc_ds2(resume=False, sampling=0, max_fields_key=2, dump_dir='output/ds2/dumps-ns-2/')
    # test_disc_ds2(resume=False, sampling=0, max_fields_key=3, dump_dir='output/ds2/dumps-ns-3/')
    # test_disc_ds2(resume=False, sampling=0, max_fields_key=4, dump_dir='output/ds2/dumps-ns-4/')
    #
    # test_disc_ds2(resume=False, sampling=5000, max_fields_key=1, dump_dir='output/ds2/dumps-s-1/')
    # test_disc_ds2(resume=False, sampling=5000, max_fields_key=2, dump_dir='output/ds2/dumps-s-2/')
    # test_disc_ds2(resume=False, sampling=5000, max_fields_key=3, dump_dir='output/ds2/dumps-s-3/')
    # test_disc_ds2(resume=False, sampling=5000, max_fields_key=4, dump_dir='output/ds2/dumps-s-4/')
    #
    # test_disc_mssql(resume=False, sampling=0, max_fields_key=1, dump_dir='output/adw/dumps-ns-1/')
    # test_disc_mssql(resume=False, sampling=0, max_fields_key=2, dump_dir='output/adw/dumps-ns-2/')
    # test_disc_mssql(resume=False, sampling=0, max_fields_key=3, dump_dir='output/adw/dumps-ns-3/')
    # test_disc_mssql(resume=False, sampling=0, max_fields_key=4, dump_dir='output/adw/dumps-ns-4/')
    #
    # test_disc_mssql(resume=False, sampling=5000, max_fields_key=1, dump_dir='output/adw/dumps-s-1/')
    # test_disc_mssql(resume=False, sampling=5000, max_fields_key=2, dump_dir='output/adw/dumps-s-2/')
    # test_disc_mssql(resume=False, sampling=5000, max_fields_key=3, dump_dir='output/adw/dumps-s-3/')
    # test_disc_mssql(resume=False, sampling=5000, max_fields_key=4, dump_dir='output/adw/dumps-s-4/')

    # test_disc_mimic(resume=False, sampling=0, max_fields_key=1, dump_dir='output/mimic/dumps-ns-1/')
    # test_disc_mimic(resume=False, sampling=0, max_fields_key=2, dump_dir='output/mimic/dumps-ns-2/')
    # test_disc_mimic(resume=False, sampling=0, max_fields_key=3, dump_dir='output/mimic/dumps-ns-3/')
    # test_disc_mimic(resume=False, sampling=0, max_fields_key=4, dump_dir='output/mimic/dumps-ns-4/')

    test_disc_mimic(resume=False, sampling=5000, max_fields_key=1, dump_dir='output/mimic/dumps-s-1/')
    test_disc_mimic(resume=False, sampling=5000, max_fields_key=2, dump_dir='output/mimic/dumps-s-2/')
    test_disc_mimic(resume=False, sampling=5000, max_fields_key=3, dump_dir='output/mimic/dumps-s-3/')
    test_disc_mimic(resume=False, sampling=5000, max_fields_key=4, dump_dir='output/mimic/dumps-s-4/')
