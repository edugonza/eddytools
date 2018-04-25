import eddytools.extraction as ex
import sqlalchemy as sq


def test_ds2():
    openslex_file_path = 'data/sample.slexmm'

    connection_params = {
        'dialect': 'postgresql',
        'username': 'ds2',
        'password': 'ds2',
        'host': 'localhost',
        'port': '5556',
        'database': 'ds2',
        'schema': 'public',
    }

    classes = None
    try:
        ex.extract_to_mm(openslex_file_path, **connection_params,
                         classes=classes, overwrite=True)
        mm_engine = ex.create_mm_engine(openslex_file_path)
        mm_conn = mm_engine.connect()
        mm_q = sq.select('*').select_from('object as O, class as CL').where('O.class_id == CL.id AND CL.name == "customers"')
        mm_res = mm_conn.execute(mm_q).fetchall()
        mm_conn.close()

        db_engine = ex.create_db_engine(**connection_params)
        db_conn = db_engine.connect()

        db_q = db_q = sq.select('*').select_from('public.customers')
        db_res = db_conn.execute(db_q).fetchall()
        db_conn.close()

        assert db_res.__len__() == mm_res.__len__()

    except Exception as e:
        raise e


def main():
    openslex_file_path = 'data/sample.slexmm'

    connection_params = {
        'dialect': 'postgresql',
        'username': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5555',
        'database': 'mimic',
        'schema': 'public',
    }

    classes = ['patients', 'admissions', 'microbiologyevents',
              'd_items']  # use this to specify a subset of the classes
    try:
        ex.extract_to_mm(openslex_file_path, **connection_params,
                         classes=classes, overwrite=True)
    except Exception as e:
        raise e


if __name__ == '__main__':
    test_ds2()
    #main()
