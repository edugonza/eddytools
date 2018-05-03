import eddytools.extraction as ex
import eddytools.schema as es
import sqlalchemy as sq
from sqlalchemy.sql.expression import text


connection_params = {
        'dialect': 'postgresql',
        'username': 'ds2',
        'password': 'ds2',
        'host': 'localhost',
        'port': '5556',
        'database': 'ds2',
        'schema': 'public',
    }

openslex_file_path = 'data/sample.slexmm'


def test_ds2():

    classes = None
    try:
        ex.extract_to_mm(openslex_file_path, **connection_params,
                         classes=classes, overwrite=True)
        assert check_mm(openslex_file_path, connection_params)
    except Exception as e:
        raise e


def check_mm(openslex_file_path, connection_params):
    mm_engine = ex.create_mm_engine(openslex_file_path)
    mm_conn = mm_engine.connect()
    mm_q = sq.select('*').select_from(text('object as O, class as CL')) \
        .where(text('O.class_id == CL.id AND CL.name == "customers"'))
    mm_res = mm_conn.execute(mm_q).fetchall()
    mm_conn.close()
    mm_engine.dispose()

    db_engine = ex.create_db_engine(**connection_params)
    db_conn = db_engine.connect()

    db_q = sq.select('*').select_from(text('public.customers'))
    db_res = db_conn.execute(db_q).fetchall()
    db_conn.close()
    db_engine.dispose()

    check = db_res.__len__() == mm_res.__len__()
    print(check)
    return check


def test_custom_metadata_extraction():
    db_engine = ex.create_db_engine(**connection_params)
    schema = connection_params['schema']
    discovered_pks = es.discover_pks(db_engine, schema=schema)
    discovered_fks = es.discover_fks(db_engine, schema=schema,
                                     pk_candidates=discovered_pks)
    db_meta = es.create_custom_metadata(connection_params, db_engine,
                                        connection_params['schema'],
                                        discovered_pks, discovered_fks)

    db_engine.dispose()
    ex.extract_to_mm(openslex_file_path, **connection_params, overwrite=True, metadata=db_meta)
    assert check_mm(openslex_file_path, connection_params)


if __name__ == '__main__':
    test_ds2()
    test_custom_metadata_extraction()
