from eddytools import extraction as ex
from eddytools import schema as es
from pprint import pprint
import json
import os


def test_disc(connection_params=None, dump_dir='data/output/dumps/', classes_for_pk=None, classes_for_fk=None):

    if connection_params is None:
        connection_params = {
            'dialect': 'postgresql',
            'username': 'ds2',
            'password': 'ds2',
            'host': 'localhost',
            'port': '5556',
            'database': 'ds2',
            'schema': 'public',
        }

    schema = connection_params['schema']
    try:

        os.makedirs(dump_dir, exist_ok=True)

        db_engine = ex.create_db_engine(**connection_params)
        metadata = ex.get_metadata(db_engine, schema=schema)
        retrieved_pks = es.retrieve_pks(metadata)
        json.dump(retrieved_pks, open('{}/{}'.format(dump_dir, 'retrieved_pks.json'), mode='wt'), indent=True)

        all_classes = es.retrieve_classes(metadata)
        classes_with_pk = retrieved_pks.keys()
        classes_without_pk = [c for c in all_classes if c not in classes_with_pk]

        if not classes_for_pk:
            classes_for_pk = all_classes

        discovered_pks = es.discover_pks(db_engine, metadata, classes_for_pk)
        json.dump(discovered_pks, open('{}/{}'.format(dump_dir, 'discovered_pks.json'), mode='wt'), indent=True)

        filtered_pks = es.filter_discovered_pks(discovered_pks, patterns=['id'])
        json.dump(filtered_pks, open('{}/{}'.format(dump_dir, 'filtered_pks.json'), mode='wt'), indent=True)

        retrieved_fks = es.retrieve_fks(metadata)
        json.dump(retrieved_fks, open('{}/{}'.format(dump_dir, 'retrieved_fks.json'), mode='wt'), indent=True)

        if not classes_for_fk:
            classes_for_fk = all_classes

        discovered_fks = es.discover_fks(db_engine, metadata, filtered_pks, classes_for_fk, max_fields_fk=4)
        json.dump(discovered_fks, open('{}/{}'.format(dump_dir, 'discovered_fks.json'), mode='wt'), indent=True)

        filtered_fks = es.filter_discovered_fks(discovered_fks, sim_threshold=0.7, topk=1)
        json.dump(filtered_fks, open('{}/{}'.format(dump_dir, 'filtered_fks.json'), mode='wt'), indent=True)

        pk_stats, pk_score = es.compute_pk_stats(all_classes, retrieved_pks, filtered_pks)
        print("\nPK stats:")
        pprint(pk_stats)
        print("\nPK score: {} ".format(pk_score))

        fk_stats, fk_score = es.compute_fk_stats(all_classes, retrieved_fks, filtered_fks)
        print("\nFK stats:")
        pprint(fk_stats)
        print("\nFK score: {} ".format(fk_score))

        return True
    except Exception as e:
        raise e


if __name__ == '__main__':
    test_disc()
