import os
import time
import sqlite3
from pkg_resources import resource_stream
import psycopg2

# SQLAlchemy imports
from sqlalchemy import create_engine
from sqlalchemy.schema import MetaData
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import types

# OpenSLEX parameters
_OPENSLEX_SCRIPT_PATH = resource_stream(__name__, 'resources/metamodel.sql')


# create a SQLite database file for the OpenSLEX mm and run the script to create all tables
def create_mm(mm_file_path, overwrite=False):
    is_success = False

    # check if file already exists
    if os.path.exists(mm_file_path):
        if overwrite:
            try:
                print("Removing OpenSLEX MM")
                os.remove(mm_file_path)
            except Exception as e:
                raise
        else:
            raise Exception("File already exists")

    # if directory doesn't exist, create directory
    openslex_dir = os.path.dirname(mm_file_path)
    if not os.path.exists(openslex_dir):
        os.makedirs(openslex_dir)

    is_connected = False

    try:
        print("Opening OpenSLEX MM")
        conn = sqlite3.connect(mm_file_path)
        is_connected = True
        cursor = conn.cursor()

        print("Reading script")
        script = _OPENSLEX_SCRIPT_PATH.read().decode()

        print("Running script")
        cursor.executescript(script)
        conn.commit()
        conn.close()
        is_connected = False
        print("OpenSLEX MM succesfully created")

    except Exception as e:
        if is_connected:
            print("Closing DB")
            conn.close()
            is_connected = False
        raise


# create engine for the OpenSLEX mm using SQLAlchemy
def create_mm_engine(openslex_file_path):
    print("Creating OpenSLEX MM engine")
    mm_url = 'sqlite:///{path}'.format(path=openslex_file_path)
    engine = create_engine(mm_url)
    print("OpenSLEX MM engine created")
    return engine


# create engine for the source database using SQLAlchemy
def create_db_engine(dialect, username, password, host, port, database, **params):
    print("Creating DB engine")
    db_url = '{dialect}://{username}:{password}@{host}:{port}/{database}'.format(
        dialect=dialect,
        username=username,
        password=password,
        host=host,
        port=port,
        database=database
    )
    engine = create_engine(db_url)
    print("DB engine created")
    return engine


# From database to metamodel

# Load the data model into the metamodel

# automap the source database into a SQLAlchemy Base object
def automap_db(db_engine, schema):
    # print("Automapping DB")
    Base = automap_base()
    Base.metadata.schema = schema
    Base.prepare(db_engine, reflect=True)
    # print("Automap finished")
    return Base, Base.metadata


# reflect the metadata of the OpenSLEX mm into a SQLAlchemy MetaData object
def get_mm_meta(mm_engine):
    print("Obtaining MM metadata")
    mm_meta = MetaData()
    mm_meta.reflect(bind=mm_engine)
    print("MM metadata obtained")
    return mm_meta


# insert values into table t using conn
def insert_values(conn, t, values):
    trans = conn.begin()
    try:
        q = t.insert().values(values)
        res = conn.execute(q)
        trans.commit()
    except:
        trans.rollback()
        raise

    return res


# get the data type of a column (integer, string, boolean, numeric, timestamp)
def get_data_type(col):
    if isinstance(col.type, types.Integer):
        return 'integer'
    elif isinstance(col.type, types.String):
        return 'string'
    elif isinstance(col.type, types.Boolean):
        return 'boolean'
    elif isinstance(col.type, types.Numeric):
        return 'numeric'
    elif isinstance(col.type, (types.Date, types.DateTime, types.Time)):
        return 'timestamp'


'''
insert the metadata of the source database (classes, attributes and relationships) into the OpenSLEX mm
returns:
class_map: mapping class_name --> class_id in the OpenSLEX mm
attr_map: mapping (class_name, attribute_name) --> attribute_id in the OpenSLEX mm
rel_map: mapping (class_name, relationship_name) --> relationship_id in the OpenSLEX mm
'''


def insert_metadata(mm_conn, mm_meta, Base, db_meta, dm_name):
    class_map = dict()
    attr_map = dict()
    rel_map = dict()

    trans = mm_conn.begin()
    try:

        dm_table = mm_meta.tables.get('datamodel')
        dm_values = {'name': dm_name}
        res_ins_dm = insert_values(mm_conn, dm_table, dm_values)
        dm_id = res_ins_dm.inserted_primary_key[0]
        db_classes = Base.classes.keys()
        for c in db_classes:
            class_table = mm_meta.tables.get('class')
            class_values = {'datamodel_id': dm_id, 'name': c}
            res_ins_class = insert_values(mm_conn, class_table, class_values)
            class_id = res_ins_class.inserted_primary_key[0]
            class_map[c] = class_id

            attrs = db_meta.tables.get('{schema}.{c}'.format(schema=db_meta.schema, c=c)).c
            for attr in attrs:
                if get_data_type(attr):
                    attr_table = mm_meta.tables.get('attribute_name')
                    attr_values = {'class_id': class_id, 'name': attr.name, 'type': get_data_type(attr)}
                    res_ins_col = insert_values(mm_conn, attr_table, attr_values)
                    attr_id = res_ins_col.inserted_primary_key[0]
                    attr_map[(c, attr.name)] = attr_id

        for c in db_classes:
            fkcs = db_meta.tables.get('{schema}.{c}'.format(schema=db_meta.schema, c=c)).foreign_key_constraints
            for fkc in fkcs:
                rel_table = mm_meta.tables.get('relationship')
                rel_values = {'source': class_map[c],
                              'target': class_map[fkc.referred_table.name],
                              'name': fkc.name}
                res_ins_rel = insert_values(mm_conn, rel_table, rel_values)
                rel_id = res_ins_rel.inserted_primary_key[0]
                rel_map[(c, fkc.name)] = rel_id

        trans.commit()
        print('transaction committed')
    except:
        trans.rollback()
        print('transaction rolled back')
        raise

    return class_map, attr_map, rel_map


# insert object, object version, object attribute values into the OpenSLEX mm for one object in the source db
def insert_object(mm_conn, obj, source_table, class_name, class_map, attr_map, rel_map, obj_v_map, mm_meta):
    trans = mm_conn.begin()
    try:
        # insert into object table
        obj_table = mm_meta.tables.get('object')
        obj_values = {'class_id': class_map[class_name]}
        res_ins_obj = insert_values(mm_conn, obj_table, obj_values)
        obj_id = res_ins_obj.inserted_primary_key[0]

        # insert into object_version table
        obj_v_table = mm_meta.tables.get('object_version')
        obj_v_values = {'object_id': obj_id, 'start_timestamp': -2, 'end_timestamp': -1}
        res_ins_obj_v = insert_values(mm_conn, obj_v_table, obj_v_values)
        obj_v_id = res_ins_obj_v.inserted_primary_key[0]
        pk_tuple = tuple(col.name for col in source_table.primary_key.columns)
        pk_values_tuple = tuple(obj[col] for col in pk_tuple)
        obj_v_map[(class_name, pk_tuple, pk_values_tuple)] = obj_v_id

        unique_constraints = [uc for uc in source_table.constraints if isinstance(uc, UniqueConstraint)]
        for uc in unique_constraints:
            unique_tuple = tuple(col.name for col in uc)
            unique_values_tuple = tuple(obj[col] for col in unique_tuple)
            obj_v_map[(class_name, unique_tuple, unique_values_tuple)] = obj_v_id

        # insert into attribute_value table
        attr_v_table = mm_meta.tables.get('attribute_value')

        attr_v_values = [{'object_version_id': obj_v_id,
                          'attribute_name_id': attr_map[(class_name, attr[0])],
                          'value': str(attr[1])
                          } for attr in obj.items() if ((class_name, attr[0] in attr_map.keys()) and attr[1])]
        res_ins_attr_v = insert_values(mm_conn, attr_v_table, attr_v_values)

        trans.commit()
    except:
        trans.rollback()
        raise


# insert all objects of one class into the OpenSLEX mm
def insert_class_objects(mm_conn, mm_meta, db_conn, db_meta, class_name, class_map, attr_map, rel_map, obj_v_map):
    print("inserting objects for class '{c}'".format(c=class_name))
    t1 = time.time()
    trans = mm_conn.begin()
    try:
        source_table = db_meta.tables.get('{s}.{c}'.format(s=db_meta.schema, c=class_name))
        objs = db_conn.execute(source_table.select())
        for obj in objs:
            insert_object(mm_conn, obj, source_table, class_name, class_map, attr_map, rel_map, obj_v_map, mm_meta)
        trans.commit()
    except:
        trans.rollback()
        raise
    print("objects for class '{c}' inserted".format(c=class_name))
    t2 = time.time()
    time_diff = t2 - t1
    print('time elapsed: {time_diff} seconds'.format(time_diff=time_diff))


# insert the relations of one object into the OpenSLEX mm
def insert_object_relations(mm_conn, mm_meta, obj, source_table, class_name, rel_map, obj_v_map):
    trans = mm_conn.begin()
    try:
        rel_table = mm_meta.tables.get('relation')
        for fkc in source_table.foreign_key_constraints:
            target_obj_v_params = (
                fkc.referred_table.name,
                tuple(fk.column.name for fk in fkc.elements),
                tuple(obj[col] for col in fkc.columns)
            )
            if target_obj_v_params in obj_v_map.keys():
                target_obj_v_id = obj_v_map[target_obj_v_params]
                source_obj_v_id = obj_v_map[(
                    source_table.name,
                    tuple(col.name for col in source_table.primary_key.columns),
                    tuple(obj[col] for col in source_table.primary_key.columns)
                )]
                rel_value = [{
                    'source_object_version_id': source_obj_v_id,
                    'target_object_version_id': target_obj_v_id,
                    'relationship_id': rel_map[(class_name, fkc.name)],
                    'start_timestamp': -2,
                    'end_timestamp': -1
                }]
                res_ins_rel = insert_values(mm_conn, rel_table, rel_value)

        trans.commit()
    except:
        trans.rollback()
        raise


# insert the relations of all objects of one class into the OpenSLEX mm
def insert_class_relations(mm_conn, mm_meta, db_conn, db_meta, class_name, rel_map, obj_v_map):
    print("inserting relations for class '{c}'".format(c=class_name))
    t1 = time.time()
    trans = mm_conn.begin()
    try:
        source_table = db_meta.tables.get('{s}.{c}'.format(s=db_meta.schema, c=class_name))
        objs = db_conn.execute(source_table.select())
        for obj in objs:
            insert_object_relations(mm_conn, mm_meta, obj, source_table, class_name, rel_map, obj_v_map)
        trans.commit()
    except:
        trans.rollback()
        raise
    print("relations for class '{c}' inserted".format(c=class_name))
    t2 = time.time()
    time_diff = t2 - t1
    print('time elapsed: {time_diff} seconds'.format(time_diff=time_diff))


# insert the objects of all classes of the source db into the OpenSLEX mm
def insert_objects(mm_conn, mm_meta, db_conn, db_meta, classes, class_map, attr_map, rel_map):
    obj_v_map = dict()
    for class_name in classes:
        insert_class_objects(mm_conn, mm_meta, db_conn, db_meta, class_name,
                             class_map, attr_map, rel_map, obj_v_map)

    for class_name in classes:
        insert_class_relations(mm_conn, mm_meta, db_conn, db_meta, class_name,
                               rel_map, obj_v_map)

    return obj_v_map


def extract_to_mm(openslex_file_path, dialect, username, password, host, port, database, schema,
                  overwrite=False, classes=None):
    # connect to the OpenSLEX mm
    try:
        create_mm(openslex_file_path, overwrite)
        mm_engine = create_mm_engine(openslex_file_path)
        db_engine = create_db_engine(dialect, username, password, host, port, database)
        Base, db_meta = automap_db(db_engine, schema)
        mm_meta = get_mm_meta(mm_engine)
        dm_name = '{database}.{schema}'.format(database=database, schema=schema)
    except Exception as e:
        print('Something went wrong: {e}'.format(e=e))
        raise e

    # insert the source's datamodel into the OpenSLEX mm
    t1 = time.time()
    mm_conn = mm_engine.connect()
    print('connection opened')
    try:
        class_map, attr_map, rel_map = insert_metadata(mm_conn, mm_meta, Base, db_meta, dm_name)
    except Exception as e:
        print('Exception: {e}'.format(e=e))
        raise e
    mm_conn.close()
    print('connection closed')
    t2 = time.time()
    time_diff = t2 - t1
    print('total time elapsed: {time_diff} seconds'.format(time_diff=time_diff))

    # insert objects into the OpenSLEX mm
    t1 = time.time()
    mm_conn = mm_engine.connect()
    db_conn = db_engine.connect()
    print('connections opened')
    try:
        if classes is None:
            classes = Base.classes.keys() # use this if you want to insert objects of all classes
        obj_v_map = insert_objects(mm_conn, mm_meta, db_conn, db_meta, classes, class_map, attr_map, rel_map)
    except Exception as e:
        print('Exception: {e}'.format(e=e))
    mm_conn.close()
    db_conn.close()
    print('connections closed')
    t2 = time.time()
    time_diff = t2 - t1
    print('total time elapsed: {time_diff} seconds'.format(time_diff=time_diff))
