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
    }

schemas = ['public']
openslex_file_path = 'output/sample.slexmm'


def test_ds2():

    classes = None
    try:
        db_engine = ex.create_db_engine(**connection_params)
        metadata = ex.get_metadata(db_engine, schemas)
        ex.extract_to_mm(openslex_file_path, connection_params, db_engine=db_engine,
                         classes=classes, overwrite=True, metadata=metadata)
        assert check_mm(openslex_file_path, connection_params, metadata=metadata)
    except Exception as e:
        raise e


def check_mm(openslex_file_path, connection_params, metadata):

    mm_engine = ex.create_mm_engine(openslex_file_path)
    mm_conn = mm_engine.connect()

    db_engine = ex.create_db_engine(**connection_params)
    db_conn = db_engine.connect()

    check = True

    for t in metadata.tables.keys():
        mm_q = sq.select([text('count(O.id) as num')]).select_from(text('object as O, class as CL')) \
            .where(text('O.class_id == CL.id and CL.name == "{}"'.format(t)))
        mm_res = mm_conn.execute(mm_q)
        mm_num = mm_res.first()
        mm_res.close()

        db_q = sq.select([text('count(*) as num')]).select_from(text(t))
        db_res = db_conn.execute(db_q)
        db_num = db_res.first()
        db_res.close()

        check_t = mm_num['num'] == db_num['num']
        check = check and check_t

        assert check

    mm_conn.close()
    mm_engine.dispose()

    db_conn.close()
    db_engine.dispose()

    return check


def test_custom_metadata_extraction():
    db_engine = ex.create_db_engine(**connection_params)
    metadata = ex.get_metadata(db_engine, schemas)
    discovered_pks = es.discover_pks(db_engine, metadata)
    discovered_fks = es.discover_fks(db_engine, metadata,
                                     pk_candidates=discovered_pks)
    db_meta = es.create_custom_metadata(db_engine, schemas,
                                        discovered_pks, discovered_fks)

    db_engine.dispose()
    ex.extract_to_mm(openslex_file_path, connection_params, overwrite=True, metadata=db_meta)
    assert check_mm(openslex_file_path, connection_params, metadata=metadata)


if __name__ == '__main__':
    test_ds2()
    #test_custom_metadata_extraction()
