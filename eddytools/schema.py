from . import extraction as ex
from sqlalchemy.engine import Connection, Engine, ResultProxy, RowProxy
from sqlalchemy.schema import MetaData, Table, Column, ForeignKeyConstraint, UniqueConstraint, PrimaryKeyConstraint
from sqlalchemy.ext.declarative.api import DeclarativeMeta
from sqlalchemy.sql.expression import select, and_, or_, text, table, func, distinct, alias
from sqlalchemy.types import _Binary, CLOB, BLOB, Text, NullType
from sqlalchemy.sql.base import ImmutableColumnCollection
import itertools
from collections import Counter
from tqdm import tqdm
import jellyfish
import numpy as np


def get_python_type(col: Column):
    default = '-'
    try:
        return col.type.python_type
    except:
        return default


def get_col_type(col: Column):
    try:
        return str(col.type)
    except:
        return None


def is_binary_type(col: Column):
    try:
        return isinstance(col.type, _Binary)
    except:
        return False


def retrieve_tables_definition(metadata: MetaData) -> dict:
    data = {}
    c: Table
    for c in metadata.tables.values():
        columns_data = []
        for col in c.columns:
            col_d = {'name': col.name}
            col_d['type'] = str(get_col_type(col))
            col_d['type_python'] = str(get_python_type(col))
            col_d['binary'] = is_binary_type(col)
            columns_data.append(col_d)
        data_c = {'name': c.name,
                  'fullname': c.fullname,
                  'schema': c.schema,
                  'columns': columns_data}
        data[c.fullname] = data_c
    return data


def filter_binary_columns(metadata: MetaData):
    c: Table
    for c in metadata.tables.values():
        cols_to_remove = [col for col in c.columns if isinstance(col.type, (_Binary, CLOB, BLOB, Text, NullType))]
        for col in cols_to_remove:
            c.columns._data.pop(col.name)


def retrieve_classes(metadata) -> list:
    classes = [c for c in metadata.tables.keys()]
    return classes


def retrieve_pks(metadata: MetaData, classes=None) -> dict:
    # Get existing PKs and UKs in schema

    pks = {}

    if classes is None:
        classes = metadata.tables.keys()
    for c in tqdm(classes):
        t: Table = metadata.tables.get(c)
        pks[t.fullname] = []
        for cons in t.constraints:
            if isinstance(cons, (UniqueConstraint, PrimaryKeyConstraint)) and cons.columns.__len__() > 0:
                pk = {
                    'table': t.name,
                    'schema': t.schema,
                    'fullname': t.fullname,
                    'pk_columns': [col.name for col in cons.columns],
                    'pk_name': cons.name,
                    'pk_columns_type': [str(get_col_type(col)) for col in cons.columns],
                }
                pks[t.fullname].append(pk)
    return pks


def check_uniqueness_comb(db_engine: Engine, metadata: MetaData, t: Table, combination: set, idx: int,
                          total_rows: int=None):
    isunique, total_rows2, unique_len = check_uniqueness(db_engine, t, combination, total_rows)
    if isunique:
        cand = {
            'table': t.name,
            'schema': t.schema,
            'fullname': t.fullname,
            'pk_name': "{}_{}_{}_pk".format(t.name, combination.__len__(), idx),
            'pk_columns': [c.name for c in combination],
            'pk_columns_type': [str(get_col_type(c)) for c in combination],
        }
    else:
        cand = {}
    return isunique, total_rows2, unique_len, cand


def check_num_comb_stats(combination: set, stats_cols: dict, total_rows: int):
    val = 1
    for col in combination:
        val = val * stats_cols[col]['num_unique_vals']
    return val >= total_rows


def get_number_of_rows(db_engine: Engine, t: Table):
    query_total = select([func.count().label('num')]).select_from(alias(t))
    res_t: ResultProxy = db_engine.execute(query_total)
    total_rows = res_t.first()['num']
    return total_rows


def discover_pks(db_engine: Engine, metadata: MetaData, classes=None, max_fields=4):
    candidates = {}
    # For each class in classes:
    # Select candidate attributes sets
    # Check uniqueness from smaller to bigger sets

    # Return smallest sets of attributes with unique values
    if classes is None:
        classes = metadata.tables.keys()
    for c in tqdm(classes, desc='Discovering PKs'):
        t: Table = metadata.tables.get(c)
        total_rows = get_number_of_rows(db_engine, t)
        stats_cols = {}
        candidates_t = []
        unique_combs = set()
        non_unique_columns = set()
        for idx, col in tqdm(enumerate(t.columns), desc='Checking unique columns'):
            isunique, num_rows, num_unique_vals, candidate = check_uniqueness_comb(db_engine, metadata, t, {col}, idx,
                                                                                   total_rows=total_rows)
            stats_cols[col] = {'isunique': isunique,
                               'num_rows': num_rows,
                               'num_unique_vals': num_unique_vals}
            if isunique:
                candidates_t.append(candidate)
                unique_combs.add(frozenset([col]))
            else:
                non_unique_columns.add(col)
        non_unique_combs = set([frozenset([col]) for col in non_unique_columns])
        for n in tqdm(range(2, min(non_unique_columns.__len__(), max_fields)+1), desc='Exploring candidates of length'):
            non_unique_combs_next = set()
            checked_comb = set()
            idx = 0
            for comb_prev in tqdm(non_unique_combs, desc='Checking combinations'):
                comb = set([col for col in comb_prev])
                non_unique_columns_aux = set([col for col in non_unique_columns if col not in comb])
                for col in non_unique_columns_aux:
                    comb_aux = set([col_comb for col_comb in comb])
                    comb_aux.add(col)
                    if comb_aux not in checked_comb:
                        checked_comb.add(frozenset(comb_aux))
                        if check_num_comb_stats(comb_aux, stats_cols, total_rows):
                            issubset = False
                            for ucomb in unique_combs:
                                if ucomb.issubset(comb_aux):
                                    issubset = True
                                    break
                            if not issubset:
                                idx = idx + 1
                                isunique, _, _, candidate = check_uniqueness_comb(db_engine, metadata, t, comb_aux, idx)
                                if isunique:
                                    candidates_t.append(candidate)
                                    unique_combs.add(frozenset(comb_aux))
                                else:
                                    non_unique_combs_next.add(frozenset(comb_aux))
            non_unique_combs = non_unique_combs_next
        candidates[c] = candidates_t
    return candidates


def filter_discovered_pks(discovered_pks: dict, patterns=['id']):
    filtered_pks = {}
    for c in discovered_pks.keys():
        filtered_pks[c] = []
        pks = discovered_pks[c]
        for pk in pks:
            for col in pk['pk_columns']:
                found = False
                if patterns:
                    for p in patterns:
                        if p in col:
                            found = True
                            break
                else:
                    found = True
                if found:
                    filtered_pks[c].append(pk)
                    break
        if filtered_pks[c].__len__() == 0:
            filtered_pks[c] = pks
    return filtered_pks


def filter_discovered_fks(discovered_fks: dict, sim_threshold=0.5, topk=3):
    filtered_fks = {}
    ranking = {}
    for c in discovered_fks.keys():
        fks = discovered_fks[c]
        ranking[c] = np.zeros(fks.__len__())
        for idx, fk in enumerate(fks):
            sim = 0
            for col, col_ref in zip(fk['fk_columns'], fk['fk_ref_columns']):
                sim += (jellyfish.jaro_distance(col, col_ref) +
                        jellyfish.jaro_distance(col, fk['fk_ref_table'])) / 2
            ranking[c][idx] = sim / fk['fk_columns'].__len__()
        maxtopk = min(topk, ranking[c].__len__())
        topk_index = np.argpartition(ranking[c], -maxtopk)[-maxtopk:]
        over_index = np.flatnonzero(ranking[c][topk_index] > sim_threshold)
        topk_fks = [fks[idx] for idx in topk_index]
        filtered_fks[c] = [topk_fks[idx] for idx in over_index]
    return filtered_fks


def check_uniqueness(db_engine: Engine, table: Table, comb, total_rows: int=None):
    if comb.__len__() == 0:
        return False
    fields = [c for c in comb]
    if not total_rows:
        total_rows = get_number_of_rows(db_engine, table)
    query_unique = select([func.count().label('num')]).select_from(alias(select(fields).distinct()))
    res_u: ResultProxy = db_engine.execute(query_unique)
    unique_len = res_u.first()['num']

    return total_rows == unique_len, total_rows, unique_len


def retrieve_fks(metadata: MetaData, classes=None) -> dict:
    # Get existing FKs in schema

    fks = {}

    if classes is None:
        classes = metadata.tables.keys()
    for c in tqdm(classes, desc='Retrieving FKs'):
        t: Table = metadata.tables.get(c)
        fks[t.fullname] = []
        k: ForeignKeyConstraint
        for k in t.foreign_key_constraints:
            fk = {
                'table': t.name,
                'schema': t.schema,
                'fullname': t.fullname,
                'fk_ref_pk': k.referred_table.primary_key.name,
                'fk_columns': [fkcol.parent.name for fkcol in k.elements],
                'fk_ref_table': k.referred_table.name,
                'fk_ref_table_fullname': k.referred_table.fullname,
                'fk_ref_columns': [fkcol.column.name for fkcol in k.elements],
                'fk_name': k.name,
                'fk_columns_type': [str(get_col_type(fkcol)) for fkcol in k.elements]
            }
            fks[t.fullname].append(fk)
    return fks


def discover_fks(db_engine: Engine, metadata: MetaData, pk_candidates, classes=None, max_fields=4):
    candidates = {}
    inclusion_cache = {}

    # For each class in classes:
    # Get candidate fields that match PKs attribute set
    # Explore pairs of PKs-FKs and check inclusion
    # Return valid pairs
    if classes is None:
        classes = metadata.tables.keys()
    for c in tqdm(classes, desc='Discovering FKs'):
        t: Table = metadata.tables.get(c)
        candidates_t = []
        for n in tqdm(range(1, min(t.columns.__len__(), max_fields)+1), desc='Exploring candidates of length'):
            combinations = itertools.combinations(t.columns, n)
            for idx_comb, comb in tqdm(enumerate(combinations), desc='Checking combinations'):
                for idx_pkcand, candidate_pk_ref in tqdm(enumerate(
                        get_candidate_pks_ref(pk_candidates,
                                              [str(get_col_type(col)) for col in comb])), desc='Checking candidates'):
                    for idx_mapping, mapping in enumerate(
                            check_inclusion(db_engine, metadata, t, comb,
                                            candidate_pk_ref, inclusion_cache)):
                        cand_fk = {
                            'table': t.name,
                            'schema': t.schema,
                            'fullname': t.fullname,
                            'fk_name': "{}_{}_{}_{}_{}_fk".format(t.name, n, idx_comb, idx_pkcand, idx_mapping),
                            'fk_ref_pk': candidate_pk_ref['pk_name'],
                            'fk_ref_table': candidate_pk_ref['table'],
                            'fk_ref_table_fullname': candidate_pk_ref['fullname'],
                            'fk_columns': [c.name for c in comb],
                            'fk_columns_type': [str(get_col_type(c)) for c in comb],
                            'fk_ref_columns': mapping,
                        }
                        candidates_t.append(cand_fk)
        candidates[c] = candidates_t
    return candidates


def check_inclusion_mem(db_engine: Engine, metadata: MetaData, table: Table, comb, candidate_pk, inclusion_cache={}):
    if comb.__len__() == 0:
        return False
    field_names_fk = [c.name for c in comb]
    field_types_fk = [str(get_col_type(c)) for c in comb]
    field_names_pk = candidate_pk['pk_columns']
    field_types_pk = candidate_pk['pk_columns_type']

    if table.fullname not in inclusion_cache:
        inclusion_cache[table.fullname] = {}
    inclusion_t = inclusion_cache[table.fullname]
    if candidate_pk['fullname'] not in inclusion_t:
        inclusion_t[candidate_pk['fullname']] = {}

    inclusion_map = inclusion_t[candidate_pk['fullname']]
    inclusion_map_for_k = {}

    for fn_fk, ft_fk in zip(field_names_fk, field_types_fk):
        if fn_fk not in inclusion_map:
            inclusion_map[fn_fk] = {}
        if fn_fk not in inclusion_map_for_k:
            inclusion_map_for_k[fn_fk] = []
        for fn_pk, ft_pk in zip(field_names_pk, field_types_pk):
            if table.fullname == candidate_pk['fullname'] and fn_fk == fn_pk:
                continue
            elif ft_fk == ft_pk:
                included = False
                if fn_pk not in inclusion_map[fn_fk]:
                    values_fk = get_values_fields(db_engine, metadata, table.fullname, [fn_fk])
                    values_pk = get_values_fields(db_engine, metadata, candidate_pk['fullname'], [fn_pk])
                    if set(values_fk).issubset(set(values_pk)):
                        included = True
                    else:
                        included = False
                else:
                    included = inclusion_map[fn_fk][fn_pk]
                inclusion_map[fn_fk][fn_pk] = included
                if included:
                    inclusion_map_for_k[fn_fk].append(fn_pk)

    possible_mappings = generate_mappings(field_names_fk, inclusion_map_for_k, [])

    valid_mappings = []

    values_fk = get_values_fields(db_engine, metadata, table.fullname, field_names_fk)
    for m in possible_mappings:
        values_pk = get_values_fields(db_engine, metadata, candidate_pk['fullname'], m)
        if set(values_fk).issubset(set(values_pk)):
            valid_mappings.append(m)

    return valid_mappings


def check_inclusion(db_engine: Engine, metadata: MetaData, table: Table, comb, candidate_pk, inclusion_cache={}):
    if comb.__len__() == 0:
        return False
    field_names_fk = [c.name for c in comb]
    field_types_fk = [str(get_col_type(c)) for c in comb]
    field_names_pk = candidate_pk['pk_columns']
    field_types_pk = candidate_pk['pk_columns_type']

    if table.fullname not in inclusion_cache:
        inclusion_cache[table.fullname] = {}
    inclusion_t = inclusion_cache[table.fullname]
    if candidate_pk['fullname'] not in inclusion_t:
        inclusion_t[candidate_pk['fullname']] = {}

    inclusion_map = inclusion_t[candidate_pk['fullname']]
    inclusion_map_for_k = {}

    for fn_fk, ft_fk in zip(field_names_fk, field_types_fk):
        if fn_fk not in inclusion_map:
            inclusion_map[fn_fk] = {}
        if fn_fk not in inclusion_map_for_k:
            inclusion_map_for_k[fn_fk] = []
        for fn_pk, ft_pk in zip(field_names_pk, field_types_pk):
            if table.fullname == candidate_pk['fullname'] and fn_fk == fn_pk:
                continue
            elif ft_fk == ft_pk:
                included = False
                if fn_pk not in inclusion_map[fn_fk]:
                    included = is_included(db_engine, metadata, table.fullname, [fn_fk], candidate_pk['fullname'], [fn_pk])
                else:
                    included = inclusion_map[fn_fk][fn_pk]
                inclusion_map[fn_fk][fn_pk] = included
                if included:
                    inclusion_map_for_k[fn_fk].append(fn_pk)

    possible_mappings = generate_mappings(field_names_fk, inclusion_map_for_k, [])

    valid_mappings = []

    # values_fk = get_values_fields(db_engine, metadata, table.fullname, field_names_fk)
    for m in tqdm(possible_mappings, desc='Checking mappings'):
        # values_pk = get_values_fields(db_engine, metadata, candidate_pk['fullname'], m)
        # if set(values_fk).issubset(set(values_pk)):
        if is_included(db_engine, metadata, table.fullname, field_names_fk, candidate_pk['fullname'], m):
            valid_mappings.append(m)

    return valid_mappings


def is_included(db_engine: Engine, metadata: MetaData, fk_tbfullname, fk_field_names, pk_tbfullname, pk_field_names):
    #query_unique = select([func.count().label('num')]).select_from(alias(select(fields).distinct()))
    tb_fk: Table = metadata.tables[fk_tbfullname]
    tb_pk: Table = metadata.tables[pk_tbfullname]
    tb_fk = tb_fk.alias('A')
    tb_pk = tb_pk.alias('B')
    fk_fields = [tb_fk.columns[col] for col in fk_field_names]
    pk_fields = [tb_pk.columns[col] for col in pk_field_names]
    query = select(fk_fields).\
        select_from(
            tb_fk.join(
                tb_pk, and_(fk_f == pk_f for fk_f, pk_f in zip(fk_fields, pk_fields)), isouter=True)).\
        where(and_(pk_f.is_(None) for pk_f in pk_fields)).limit(1)
    #print(query)
    res: ResultProxy = db_engine.execute(query)
    first_res = res.first()
    not_included = first_res is None
    # values_fk = get_values_fields(db_engine, metadata, fk_tbfullname, fk_fields)
    # values_pk = get_values_fields(db_engine, metadata, pk_tbfullname, pk_fields)
    # return set(values_fk).issubset(set(values_pk))
    return not_included


def generate_mappings(field_names_fk, inclusion_map, mapping):
    if field_names_fk.__len__() == 0:
        return [mapping]
    mappings = []
    field_names_fk_rest = field_names_fk[1:]
    fn_tk = field_names_fk[0]
    for inc_fn_pk in inclusion_map[fn_tk]:
        if inc_fn_pk not in mapping:
            mappings.extend(generate_mappings(field_names_fk_rest, inclusion_map, mapping+[inc_fn_pk]))
    return mappings


def get_values_fields(db_engine: Engine, metadata: MetaData, table: str, fields: list):
    tb = metadata.tables.get(table)
    query = select([tb.columns.get(f) for f in fields]).select_from(tb)
    res: ResultProxy = db_engine.execute(query)
    values = []
    for r in res:
        values.append(tuple(r.values()))
    return values


def get_candidate_pks_ref(pks: dict, types: list):
    candidate_pks = []
    for tn in pks.keys():
        for pk in pks[tn]:
            if Counter([type for type in types]) == Counter([type for type in pk['pk_columns_type']]):
                candidate_pks.append(pk)
    return candidate_pks


def prune_pks_with_fks(pks: dict, fks: dict):
    pruned_pks = {}
    for c in fks.keys():
        for fk in fks[c]:
            c_pk_ref = fk['fk_ref_table_fullname']
            pk_name = fk['fk_ref_pk']
            pks_c = pks[c_pk_ref]
            if pks_c.__len__() > 1:
                if c_pk_ref not in pruned_pks:
                    pruned_pks[c_pk_ref] = []
                for pk in pks_c:
                    if pk['pk_name'] == pk_name:
                        if pk not in pruned_pks[c_pk_ref]:
                            pruned_pks[c_pk_ref].append(pk)
                        break
            else:
                pruned_pks[c_pk_ref] = pks_c

    for c in pks.keys():
        if c not in pruned_pks:
            pruned_pks[c] = pks[c]
    return pruned_pks


def compute_pk_stats(all_classes, pks: dict, discovered_pks: dict):
    stats = {}
    scores = {}
    for c in all_classes:
        true_pks_c = pks.get(c) if pks.get(c) else []
        disc_pks_c = discovered_pks.get(c) if discovered_pks.get(c) else []
        disc_tp = [0 for i in disc_pks_c]
        for true_pk in true_pks_c:
            for disc_idx, disc_pk in enumerate(disc_pks_c):
                if disc_pk['pk_columns'] == true_pk['pk_columns']:
                    disc_tp[disc_idx] = 1
        stats[c] = {'tp': sum(disc_tp),
                    'fp': disc_tp.__len__() - sum(disc_tp),
                    'p': true_pks_c.__len__(),
                    }
        stats[c]['prec'] = precision(stats[c]['tp'], stats[c]['fp'])
        stats[c]['rec'] = recall(stats[c]['tp'], stats[c]['p'])
        scores = {'precision': np.mean([stats[c]['prec'] for c in stats.keys()]),
                  'recall': np.mean([stats[c]['rec'] for c in stats.keys()])}
    return stats, scores


def compute_fk_stats(all_classes, fks: dict, discovered_fks: dict):
    stats = {}
    scores = {}
    for c in all_classes:
        true_fks_c = fks.get(c) if fks.get(c) else []
        disc_fks_c = discovered_fks.get(c) if discovered_fks.get(c) else []
        disc_tp = [0 for i in disc_fks_c]
        for true_fk in true_fks_c:
            for disc_idx, disc_fk in enumerate(disc_fks_c):
                if disc_fk['fk_ref_table_fullname'] == true_fk['fk_ref_table_fullname']:
                    if disc_fk['fk_columns'] == true_fk['fk_columns']:
                        if disc_fk['fk_ref_columns'] == true_fk['fk_ref_columns']:
                            disc_tp[disc_idx] = 1
        stats[c] = {'tp': sum(disc_tp),
                    'fp': disc_tp.__len__() - sum(disc_tp),
                    'p': true_fks_c.__len__(),
                    }
        stats[c]['prec'] = precision(stats[c]['tp'], stats[c]['fp'])
        stats[c]['rec'] = recall(stats[c]['tp'], stats[c]['p'])
        scores = {'precision': np.mean([stats[c]['prec'] for c in stats.keys()]),
                  'recall': np.mean([stats[c]['rec'] for c in stats.keys()])}
    return stats, scores


def precision(tp: int, fp: int):
    if tp > 0 or fp > 0:
        return tp / (tp + fp)
    else:
        return 1


def recall(tp: int, p: int):
    if tp > 0 or p > 0:
        return tp / p
    else:
        return 1


def create_custom_metadata(db_engine: Engine, schema: str, pks: dict, fks: dict):
    metadata = MetaData(bind=db_engine)
    metadata.reflect(schema=schema)

    # Discard any existing pk, uk or fk
    t: Table
    for t in metadata.tables.values():
        t.constraints.clear()
        t.primary_key = None
        t.foreign_keys.clear()

    # Add custom pk, uk, and fk
    for c in pks:
        t: Table = metadata.tables.get(c)
        for idx, k in enumerate(pks[c]):
            if idx == 0:
                uk = PrimaryKeyConstraint(*k['pk_columns'])
            else:
                uk = UniqueConstraint(*k['pk_columns'])
            t.append_constraint(uk)

    for c in fks:
        t: Table = metadata.tables.get(c)
        for k in fks[c]:
            refcolumns = ['{}.{}'.format(k['fk_ref_table_fullname'], col) for col in k['fk_ref_columns']]
            fk = ForeignKeyConstraint(columns=k['fk_columns'],
                                      refcolumns=refcolumns,
                                      name=k['fk_name'])
            t.append_constraint(fk)

    return metadata
