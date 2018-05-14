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

    es.full_discovery(connection_params, schemas=schemas, dump_dir='output/ds2/dumps/', max_fields_key=4)


if __name__ == '__main__':
    test_disc_ds2()

