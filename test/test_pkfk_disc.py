from eddytools import schema as es


def test_disc_ds2():

    connection_params = {
        'dialect': 'postgresql',
        'username': 'ds2',
        'password': 'ds2',
        'host': 'localhost',
        'port': '5556',
        'database': 'ds2',
    }

    schemas = ['public']

    es.full_discovery(connection_params, schemas=schemas, max_fields_key=4)


def test_disc_mssql():

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

    return es.full_discovery(connection_params, schemas=None, dump_dir='data/output/adw2/dumps/', max_fields_key=2)


if __name__ == '__main__':
    #test_disc_ds2()
    test_disc_mssql()

