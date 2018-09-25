from eddytools.extraction import get_data_type
import pickle
from sqlalchemy.schema import MetaData, Table, Column,\
    Constraint, PrimaryKeyConstraint, UniqueConstraint, ForeignKeyConstraint
import numpy as np


def test_db_schema_stats(meta_path):
    meta: MetaData = pickle.load(open(meta_path, mode='rb'))

    colcount = []
    tscolcount = []
    pkcount = []
    ukcount = []
    fkcount = []
    table_names = []

    for tname in meta.tables:
        t: Table = meta.tables[tname]
        table_names.append(tname)
        colcount.append(t.columns.__len__())
        tscolcount.append(sum([1 for c in t.columns if get_data_type(c) == 'timestamp']))
        pkcount.append(1 if t.primary_key else 0)
        ukcount.append(sum([1 for c in t.constraints if type(c) == UniqueConstraint]))
        fkcount.append(sum([1 for c in t.constraints if type(c) == ForeignKeyConstraint]))
        print('{}: {}'.format(t.fullname, t.columns.__len__()))

    print('# of tables: {}'.format(colcount.__len__()))
    print('# of columns: {}'.format(sum(colcount)))
    print('# of timestamp columns: {}'.format(sum(tscolcount)))
    print('Mean cols per table: {}'.format(np.mean(colcount)))
    print('Median cols per table: {}'.format(np.median(colcount)))
    print('Mean timestamp cols per table: {}'.format(np.mean(tscolcount)))
    print('Median timestamp cols per table: {}'.format(np.median(tscolcount)))
    print('# of pks: {}'.format(sum(pkcount)))
    print('# of uks: {}'.format(sum(ukcount)))
    print('# of fks: {}'.format(sum(fkcount)))
    print('Mean uks per table: {}'.format(np.mean(ukcount)))
    print('Median uks per table: {}'.format(np.median(ukcount)))
    print('Mean fks per table: {}'.format(np.mean(fkcount)))
    print('Median fks per table: {}'.format(np.median(fkcount)))


if __name__ == '__main__':
    dir = 'output/adw/dumps/metadata.pickle'

    import sys
    if sys.argv.__len__() > 1:
        dir = sys.argv[1]

    test_db_schema_stats(meta_path=dir)
