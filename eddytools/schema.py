from . import extraction as ex
from sqlalchemy.engine import Connection, Engine, ResultProxy, RowProxy
from sqlalchemy.schema import MetaData, Table, Column, ForeignKeyConstraint, UniqueConstraint, PrimaryKeyConstraint
from sqlalchemy.ext.declarative.api import DeclarativeMeta
from sqlalchemy.sql.expression import select, and_, or_, text
import itertools
from collections import Counter
from tqdm import tqdm
import jellyfish
import numpy as np


def retrieve_classes(db_engine: Engine, schema) -> list:
    base: DeclarativeMeta
    base_metadata: MetaData
    base, base_metadata = ex.automap_db(db_engine, schema)
    classes = [c for c in base_metadata.tables.keys()]
    return classes


def retrieve_pks(db_engine: Engine, schema, classes=None) -> dict:
    base: DeclarativeMeta
    base_metadata: MetaData
    base, base_metadata = ex.automap_db(db_engine, schema)
    # Get existing PKs and UKs in schema

    pks = {}

    if classes is None:
        classes = base_metadata.tables.keys()
    for c in tqdm(classes):
        t: Table = base_metadata.tables.get(c)
        pks[t.fullname] = []
        for cons in t.constraints:
            if isinstance(cons, (UniqueConstraint, PrimaryKeyConstraint)) and cons.columns.__len__() > 0:
                pk = {
                    'table': t.name,
                    'schema': t.schema,
                    'fullname': t.fullname,
                    'pk_columns': [col.name for col in cons.columns],
                    'pk_name': cons.name,
                    'pk_columns_type': [str(col.type) for col in cons.columns],
                }
                pks[t.fullname].append(pk)
    return pks


def discover_pks(db_engine: Engine, schema, classes):
    base: DeclarativeMeta
    base_metadata: MetaData
    base, base_metadata = ex.automap_db(db_engine, schema)

    candidates = {}
    # For each class in classes:
    # Select candidate attributes sets
    # Check uniqueness from smaller to bigger sets

    # Return smallest sets of attributes with unique values
    if classes is None:
        classes = base_metadata.tables.keys()
    for c in tqdm(classes, desc='Discovering PKs'):
        t: Table = base_metadata.tables.get(c)
        candidates_t = []
        for n in range(t.columns.__len__()):
            if candidates_t.__len__() > 0:
                break
            combinations = itertools.combinations(t.columns, n)
            for idx, comb in enumerate(combinations):
                if check_uniqueness(db_engine, t, comb):
                    cand = {
                        'table': t.name,
                        'schema': t.schema,
                        'fullname': t.fullname,
                        'pk_name': "{}_{}_{}_pk".format(t.name, n, idx),
                        'pk_columns': [c.name for c in comb],
                        'pk_columns_type': [str(c.type) for c in comb],
                    }
                    candidates_t.append(cand)
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
                for p in patterns:
                    if p in col:
                        found = True
                        break
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


def check_uniqueness(db_engine: Engine, table: Table, comb):
    if comb.__len__() == 0:
        return False
    fields = [text(c.name) for c in comb]
    query = select(fields).select_from(table)
    res: ResultProxy = db_engine.execute(query)
    values = []
    for r in res:
        values.append(r.values())

    total_len = values.__len__()
    unique_len = set(tuple(i) for i in values).__len__()

    return total_len == unique_len


def retrieve_fks(db_engine: Engine, schema, classes=None) -> dict:
    base: DeclarativeMeta
    base_metadata: MetaData
    base, base_metadata = ex.automap_db(db_engine, schema)
    # Get existing FKs in schema

    fks = {}

    if classes is None:
        classes = base_metadata.tables.keys()
    for c in tqdm(classes, desc='Retrieving FKs'):
        t: Table = base_metadata.tables.get(c)
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
                'fk_columns_type': [str(fkcol.parent.type) for fkcol in k.elements]
            }
            fks[t.fullname].append(fk)
    return fks


def discover_fks(db_engine: Engine, schema, pk_candidates, classes, max_fields_fk):
    base: DeclarativeMeta
    base_metadata: MetaData
    base, base_metadata = ex.automap_db(db_engine, schema)

    candidates = {}
    inclusion_cache = {}

    # For each class in classes:
    # Get candidate fields that match PKs attribute set
    # Explore pairs of PKs-FKs and check inclusion
    # Return valid pairs
    if classes is None:
        classes = base_metadata.tables.keys()
    for c in tqdm(classes, desc='Discovering FKs'):
        t: Table = base_metadata.tables.get(c)
        candidates_t = []
        for n in tqdm(range(1, min(t.columns.__len__(), max_fields_fk)+1), desc='Exploring candidates of length'):
            combinations = itertools.combinations(t.columns, n)
            for idx, comb in tqdm(enumerate(combinations), desc='Checking combinations'):
                for candidate_pk_ref in get_candidate_pks_ref(pk_candidates, [str(col.type) for col in comb]):
                    for mapping in check_inclusion(db_engine, t, comb, candidate_pk_ref, inclusion_cache):
                        cand_fk = {
                            'table': t.name,
                            'schema': t.schema,
                            'fullname': t.fullname,
                            'fk_name': "{}_{}_{}_fk".format(t.name, n, idx),
                            'fk_ref_pk': candidate_pk_ref['pk_name'],
                            'fk_ref_table': candidate_pk_ref['table'],
                            'fk_ref_table_fullname': candidate_pk_ref['fullname'],
                            'fk_columns': [c.name for c in comb],
                            'fk_columns_type': [str(c.type) for c in comb],
                            'fk_ref_columns': mapping,
                        }
                        candidates_t.append(cand_fk)
        candidates[c] = candidates_t
    return candidates


def check_inclusion(db_engine: Engine, table: Table, comb, candidate_pk, inclusion_cache={}):
    if comb.__len__() == 0:
        return False
    field_names_fk = [c.name for c in comb]
    field_types_fk = [str(c.type) for c in comb]
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
                    values_fk = get_values_fields(db_engine, table.fullname, [fn_fk])
                    values_pk = get_values_fields(db_engine, candidate_pk['fullname'], [fn_pk])
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

    values_fk = get_values_fields(db_engine, table.fullname, field_names_fk)
    for m in possible_mappings:
        values_pk = get_values_fields(db_engine, candidate_pk['fullname'], m)
        if set(values_fk).issubset(set(values_pk)):
            valid_mappings.append(m)

    return valid_mappings


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


def get_values_fields(db_engine: Engine, table: str, fields: list):
    query = select([text(f) for f in fields]).select_from(text(table))
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


def compute_pk_stats(all_classes, pks: dict, discovered_pks: dict):
    stats = {}
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
