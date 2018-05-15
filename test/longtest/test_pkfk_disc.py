from eddytools import schema as es


def test_disc_mssql(resume=False):

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

    return es.full_discovery(connection_params, schemas=None, dump_dir='output/adw/dumps/', max_fields_key=2,
                             resume=resume)


if __name__ == '__main__':
    test_disc_mssql(resume=False)

