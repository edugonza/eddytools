from datetime import datetime
from math import log1p, sqrt
import numpy as np
from sqlalchemy import select as select, func, and_, distinct
import string


################################  utils


def _check_already_calculated(feature_name, feature_values):
    for fv in feature_values:
        if feature_name not in fv:
            return False
    return True


################################  general


def candidate_type(candidates, feature_values, engine, meta):
    if _check_already_calculated('candidate_type', feature_values):
        return
    for c, fv in zip(candidates, feature_values):
        if c.activity_identifier_attribute_id is None:
            fv['candidate_type'] = 'column'
        elif c.relationship_id is None:
            fv['candidate_type'] = 'in-table'
        else:
            fv['candidate_type'] = 'lookup'


def data_type(candidates, feature_values, engine, meta):
    if _check_already_calculated('data_type', feature_values):
        return
    attr_ids = {c.activity_identifier_attribute_id for c in candidates}
    t = meta.tables.get('attribute_name')
    q = (
        select([t.c.id, t.c.type])
        .where(t.c.id.in_(attr_ids))
    )
    result = engine.execute(q)
    result_dict = {row['id']: row['type'] for row in result}
    for c, fv in zip(candidates, feature_values):
        fv['data_type'] = result_dict.get(c.activity_identifier_attribute_id)


def timestamp_attribute_id(candidates, feature_values, engine, meta):
    if _check_already_calculated('timestamp_attribute_id', feature_values):
        return
    for c, fv in zip(candidates, feature_values):
        fv['timestamp_attribute_id'] = c.timestamp_attribute_id


def activity_identifier_attribute_id(candidates, feature_values, engine, meta):
    if _check_already_calculated('activity_identifier_attribute_id', feature_values):
        return
    for c, fv in zip(candidates, feature_values):
        fv['activity_identifier_attribute_id'] = c.activity_identifier_attribute_id


def relationship_id(candidates, feature_values, engine, meta):
    if _check_already_calculated('relationship_id', feature_values):
        return
    for c, fv in zip(candidates, feature_values):
        fv['relationship_id'] = c.relationship_id


################################  row counts


def nr_timestamps(candidates, feature_values, engine, meta):
    if _check_already_calculated('nr_timestamps', feature_values):
        return
    ts_ids = {c.timestamp_attribute_id for c in candidates}
    t = meta.tables.get('attribute_value')
    q = (select([t.c.attribute_name_id, func.count(t.c.value).label('nr_values')])
         .where(t.c.attribute_name_id.in_(ts_ids))
         .group_by(t.c.attribute_name_id)
         )
    result = engine.execute(q)
    result_dict = {row['attribute_name_id']: row['nr_values'] for row in result}
    for c, fv in zip(candidates, feature_values):
        fv['nr_timestamps'] = result_dict.get(c.timestamp_attribute_id, 0)


def nr_values_where_timestamp(candidates, feature_values, engine, meta):
    if _check_already_calculated('nr_values_where_timestamp', feature_values):
        return
    starttime = datetime.now()

    t1 = meta.tables.get('attribute_value').alias()
    t2 = meta.tables.get('attribute_value').alias()
    t_rels = meta.tables.get('relation')

    for c, fv in zip(candidates, feature_values):
        if 'nr_timestamps' not in fv:
            nr_timestamps(candidates, feature_values, engine, meta)
        if not c.activity_identifier_attribute_id:
            fv['nr_values_where_timestamp'] = fv['nr_timestamps']
            fv['nr_unique_values_where_timestamp'] = 1

    # in-table
    ts_ids = {c.timestamp_attribute_id for c in candidates if c.activity_identifier_attribute_id
                and not c.relationship_id}
    a_id_ids = {c.activity_identifier_attribute_id for c in candidates if c.activity_identifier_attribute_id
                and not c.relationship_id}
    q = (
        select([t1.c.attribute_name_id.label('ts_id'),
                t2.c.attribute_name_id.label('a_id_id'),
                func.count(t2.c.value).label('nr_values_where_timestamp'),
                func.count(distinct(t2.c.value)).label('nr_unique_values_where_timestamp')])
            .select_from(t1.join(t2, t1.c.object_version_id == t2.c.object_version_id))
            .where(and_(t1.c.attribute_name_id.in_(ts_ids),
                        t2.c.attribute_name_id.in_(a_id_ids),
                        ))
            .group_by(t1.c.attribute_name_id, t2.c.attribute_name_id)
    )
    res = engine.execute(q)
    for row in res:
        for c, fv in zip(candidates, feature_values):
            if (c.timestamp_attribute_id == row['ts_id']
                    and c.activity_identifier_attribute_id == row['a_id_id']
                    and not c.relationship_id):
                fv['nr_values_where_timestamp'] = row['nr_values_where_timestamp']
                fv['nr_unique_values_where_timestamp'] = row['nr_unique_values_where_timestamp']

    # lookup
    ts_ids = {c.timestamp_attribute_id for c in candidates if c.relationship_id}
    a_id_ids = {c.activity_identifier_attribute_id for c in candidates if c.relationship_id}
    rel_ids = {c.relationship_id for c in candidates if c.relationship_id}
    q = (
        select([t1.c.attribute_name_id.label('ts_id'),
                t2.c.attribute_name_id.label('a_id_id'),
                t_rels.c.relationship_id.label('rel_id'),
                func.count(t2.c.value).label('nr_values_where_timestamp'),
                func.count(distinct(t2.c.value)).label('nr_unique_values_where_timestamp')])
            .select_from(t1
                         .join(t_rels, t1.c.object_version_id == t_rels.c.source_object_version_id)
                         .join(t2, t_rels.c.target_object_version_id == t2.c.object_version_id))
            .where(and_(t1.c.attribute_name_id.in_(ts_ids),
                        t_rels.c.relationship_id.in_(rel_ids),
                        t2.c.attribute_name_id.in_(a_id_ids),
                        ))
            .group_by(t1.c.attribute_name_id, t2.c.attribute_name_id, t_rels.c.relationship_id)
    )
    res = engine.execute(q)
    for row in res:
        for c, fv in zip(candidates, feature_values):
            if (c.timestamp_attribute_id == row['ts_id']
                    and c.activity_identifier_attribute_id == row['a_id_id']
                    and c.relationship_id == row['rel_id']):
                fv['nr_values_where_timestamp'] = row['nr_values_where_timestamp']
                fv['nr_unique_values_where_timestamp'] = row['nr_unique_values_where_timestamp']

    for fv in feature_values:
        if 'nr_values_where_timestamp' not in fv:
            fv['nr_values_where_timestamp'] = 0
            fv['nr_unique_values_where_timestamp'] = 0

    endtime = datetime.now()
    print("time for 'nr_values_where_timestamp': " + str(endtime - starttime))


def nr_unique_values_where_timestamp(candidates, feature_values, engine, meta):
    if _check_already_calculated('nr_unique_values_where_timestamp', feature_values):
        return
    nr_values_where_timestamp(candidates, feature_values, engine, meta)


def not_null_ratio(candidates, feature_values, engine, meta):
    if _check_already_calculated('not_null_ratio', feature_values):
        return
    for fv in feature_values:
        if 'nr_values_where_timestamp' not in fv:
            nr_values_where_timestamp(candidates, feature_values, engine, meta)
        if 'nr_timestamps' not in fv:
            nr_timestamps(candidates, feature_values, engine, meta)
        if fv['nr_timestamps'] == 0:
            fv['not_null_ratio'] = None
        else:
            fv['not_null_ratio'] = fv['nr_values_where_timestamp'] / fv['nr_timestamps']


def log_nr_timestamps(candidates, feature_values, engine, meta):
    if _check_already_calculated('log_nr_timestamps', feature_values):
        return
    for fv in feature_values:
        if 'nr_timestamps' not in fv:
            nr_timestamps(candidates, feature_values, engine, meta)
        fv['log_nr_timestamps'] = log1p(fv['nr_timestamps'])


def log_nr_unique_values_where_timestamp(candidates, feature_values, engine, meta):
    if _check_already_calculated('log_nr_unique_values_where_timestamp', feature_values):
        return
    for fv in feature_values:
        if 'nr_unique_values_where_timestamp' not in fv:
            nr_unique_values_where_timestamp(candidates, feature_values, engine, meta)
        fv['log_nr_unique_values_where_timestamp'] = log1p(fv['nr_unique_values_where_timestamp'])


def uniqueness_ratio(candidates, feature_values, engine, meta):
    if _check_already_calculated('uniqueness_ratio', feature_values):
        return
    for fv in feature_values:
        if 'nr_unique_values_where_timestamp' not in fv:
            nr_unique_values_where_timestamp(candidates, feature_values, engine, meta)
        if 'nr_timestamps' not in fv:
            nr_timestamps(candidates, feature_values, engine, meta)
        if fv['nr_timestamps'] == 0:
            fv['uniqueness_ratio'] = None
        else:
            fv['uniqueness_ratio'] = fv['nr_unique_values_where_timestamp'] / fv['nr_timestamps']


def log_uniqueness_ratio(candidates, feature_values, engine, meta):
    if _check_already_calculated('log_uniqueness_ratio', feature_values):
        return
    for fv in feature_values:
        if 'log_nr_unique_values_where_timestamp' not in fv:
            log_nr_unique_values_where_timestamp(candidates, feature_values, engine, meta)
        if 'log_nr_timestamps' not in fv:
            log_nr_timestamps(candidates, feature_values, engine, meta)
        if fv['log_nr_timestamps'] == 0:
            fv['log_uniqueness_ratio'] = 0
        else:
            fv['log_uniqueness_ratio'] = fv['log_nr_unique_values_where_timestamp'] / fv['log_nr_timestamps']


def nr_no_timestamp(candidates, feature_values, engine, meta):
    if _check_already_calculated('nr_no_timestamp', feature_values):
        return
    ts_ids = {c.timestamp_attribute_id for c in candidates}
    t_class = meta.tables.get('class')
    t_attr = meta.tables.get('attribute_name')
    t_obj = meta.tables.get('object')
    t_obj_v = meta.tables.get('object_version')
    t_attr_v = meta.tables.get('attribute_value')
    q = (
        select([
            t_attr.c.id.label('ts_id'),
            func.count(t_obj_v.c.id).label('nr_no_timestamp'),
        ])
        .select_from(
            t_class
            .join(t_attr, t_class.c.id == t_attr.c.class_id)
            .join(t_obj, t_class.c.id == t_obj.c.class_id)
            .join(t_obj_v, t_obj.c.id == t_obj_v.c.object_id)
            .outerjoin(t_attr_v, and_(t_attr.c.id == t_attr_v.c.attribute_name_id,
                                      t_obj_v.c.id == t_attr_v.c.object_version_id)))
        .where(and_(
            t_attr.c.id.in_(ts_ids),
            t_attr_v.c.id.is_(None)
        ))
        .group_by(t_attr.c.id)
    )
    result = engine.execute(q)
    result_dict = {row['ts_id']: row['nr_no_timestamp'] for row in result}
    for c, fv in zip(candidates, feature_values):
        fv['nr_no_timestamp'] = result_dict.get(c.timestamp_attribute_id, 0)


def timestamp_ratio(candidates, feature_values, engine, meta):
    if _check_already_calculated('timestamp_ratio', feature_values):
        return
    for c, fv in zip(candidates, feature_values):
        if 'nr_timestamps' not in fv:
            nr_timestamps(candidates, feature_values, engine, meta)
        if 'nr_no_timestamp' not in fv:
            nr_no_timestamp(candidates, feature_values, engine, meta)
        ts = fv['nr_timestamps']
        no_ts = fv['nr_no_timestamp']
        if (ts + no_ts) == 0:
            fv['timestamp_ratio'] = 0
        else:
            fv['timestamp_ratio'] = ts / (ts + no_ts)


def nr_values_where_no_timestamp(candidates, feature_values, engine, meta):
    if _check_already_calculated('nr_values_where_no_timestamp', feature_values):
        return

    t_class = meta.tables.get('class')
    t_obj = meta.tables.get('object')
    t_obj_v = meta.tables.get('object_version')
    t_attr_1 = meta.tables.get('attribute_name').alias()
    t_attr_v_1 = meta.tables.get('attribute_value').alias()
    t_attr_v_2 = meta.tables.get('attribute_value').alias()
    t_rels = meta.tables.get('relation')

    # in-table
    ts_ids = {c.timestamp_attribute_id for c in candidates if c.activity_identifier_attribute_id
              and not c.relationship_id}
    a_id_ids = {c.activity_identifier_attribute_id for c in candidates if c.activity_identifier_attribute_id
                and not c.relationship_id}
    q = (
        select([
            t_attr_1.c.id.label('ts_id'),
            t_attr_v_2.c.attribute_name_id.label('a_id_id'),
            func.count(t_attr_v_2.c.id).label('nr_values_where_no_timestamp')
        ])
        .select_from(t_class
                     .join(t_attr_1, t_class.c.id == t_attr_1.c.class_id)
                     .join(t_obj, t_class.c.id == t_obj.c.class_id)
                     .join(t_obj_v, t_obj.c.id == t_obj_v.c.object_id)
                     .outerjoin(t_attr_v_1, and_(t_attr_1.c.id == t_attr_v_1.c.attribute_name_id,
                                                 t_obj_v.c.id == t_attr_v_1.c.object_version_id))
                     .join(t_attr_v_2, t_obj_v.c.id == t_attr_v_2.c.object_version_id))
        .where(and_(
            t_attr_1.c.id.in_(ts_ids),
            t_attr_v_1.c.id.is_(None),
            t_attr_v_2.c.attribute_name_id.in_(a_id_ids),
        ))
        .group_by(t_attr_1.c.id, t_attr_v_2.c.attribute_name_id)
    )
    result = engine.execute(q)
    result_dict = {(row['ts_id'], row['a_id_id']): row['nr_values_where_no_timestamp'] for row in result}
    for c, fv in zip(candidates, feature_values):
        if c.activity_identifier_attribute_id and not c.relationship_id:
            fv['nr_values_where_no_timestamp'] = result_dict.get(
                (c.timestamp_attribute_id, c.activity_identifier_attribute_id), 0)

    # lookup
    ts_ids = {c.timestamp_attribute_id for c in candidates if c.relationship_id}
    a_id_ids = {c.activity_identifier_attribute_id for c in candidates if c.relationship_id}
    rel_ids = {c.relationship_id for c in candidates if c.relationship_id}
    q = (
        select([
            t_attr_1.c.id.label('ts_id'),
            t_attr_v_2.c.attribute_name_id.label('a_id_id'),
            t_rels.c.relationship_id.label('rel_id'),
            func.count(t_attr_v_2.c.id).label('nr_values_where_no_timestamp')
        ])
        .select_from(t_class
                     .join(t_attr_1, t_class.c.id == t_attr_1.c.class_id)
                     .join(t_obj, t_class.c.id == t_obj.c.class_id)
                     .join(t_obj_v, t_obj.c.id == t_obj_v.c.object_id)
                     .outerjoin(t_attr_v_1, and_(t_attr_1.c.id == t_attr_v_1.c.attribute_name_id,
                                            t_obj_v.c.id == t_attr_v_1.c.object_version_id))
                     .join(t_rels, t_obj_v.c.id == t_rels.c.source_object_version_id)
                     .join(t_attr_v_2, t_rels.c.target_object_version_id == t_attr_v_2.c.object_version_id))
        .where(and_(
            t_attr_1.c.id.in_(ts_ids),
            t_attr_v_1.c.id.is_(None),
            t_rels.c.relationship_id.in_(rel_ids),
            t_attr_v_2.c.attribute_name_id.in_(a_id_ids),
        ))
        .group_by(t_attr_1.c.id, t_attr_v_2.c.attribute_name_id, t_rels.c.relationship_id)
    )
    result = engine.execute(q)
    result_dict = {(row['ts_id'], row['a_id_id'], row['rel_id']):
                       row['nr_values_where_no_timestamp'] for row in result}
    for c, fv in zip(candidates, feature_values):
        if c.relationship_id:
            fv['nr_values_where_no_timestamp'] = result_dict.get(
                (c.timestamp_attribute_id, c.activity_identifier_attribute_id, c.relationship_id), 0)

    # column
    for fv in feature_values:
        if 'nr_values_where_no_timestamp' not in fv:
            fv['nr_values_where_no_timestamp'] = 0


def no_ts_no_value_ratio(candidates, feature_values, engine, meta):
    if _check_already_calculated('no_ts_no_value_ratio', feature_values):
        return
    for c, fv in zip(candidates, feature_values):
        if 'nr_no_timestamp' not in fv:
            nr_no_timestamp(candidates, feature_values, engine, meta)
        if 'nr_values_where_no_timestamp' not in fv:
            nr_values_where_no_timestamp(candidates, feature_values, engine, meta)
        no_ts = fv['nr_no_timestamp']
        value_no_ts = fv['nr_values_where_no_timestamp']
        if no_ts == 0:
            fv['no_ts_no_value_ratio'] = 0
        else:
            fv['no_ts_no_value_ratio'] = (no_ts - value_no_ts) / no_ts


################################  text


def text_length_mean(candidates, feature_values, engine, meta):
    text_length_mean_std_cv(candidates, feature_values, engine, meta)


def log_text_length_mean(candidates, feature_values, engine, meta):
    if _check_already_calculated('log_text_length_mean', feature_values):
        return
    for fv in feature_values:
        if 'text_length_mean' not in fv:
            nr_timestamps(candidates, feature_values, engine, meta)
        fv['log_text_length_mean'] = log1p(fv['text_length_mean'])


def text_length_std(candidates, feature_values, engine, meta):
    text_length_mean_std_cv(candidates, feature_values, engine, meta)


def text_length_cv(candidates, feature_values, engine, meta):
    text_length_mean_std_cv(candidates, feature_values, engine, meta)


def text_length_mean_std_cv(candidates, feature_values, engine, meta):
    if (_check_already_calculated('text_length_mean', feature_values)
            and _check_already_calculated('text_length_std', feature_values)
            and _check_already_calculated('text_length_cv', feature_values)):
        return

    for fv in feature_values:
        if 'data_type' not in fv:
            data_type(candidates, feature_values, engine, meta)

    t1 = meta.tables.get('attribute_value').alias()
    t2 = meta.tables.get('attribute_value').alias()
    t_rels = meta.tables.get('relation')

    # in-table
    ts_ids = {c.timestamp_attribute_id for c, fv in zip(candidates, feature_values)
              if (c.activity_identifier_attribute_id and not c.relationship_id)
              and fv['data_type'] == 'string'}
    a_id_ids = {c.activity_identifier_attribute_id for c, fv in zip(candidates, feature_values)
                if (c.activity_identifier_attribute_id and not c.relationship_id)
                and fv['data_type'] == 'string'}
    q = (
        select([t1.c.attribute_name_id.label('ts_id'),
                t2.c.attribute_name_id.label('a_id_id'),
                t2.c.value])
            .select_from(t1.join(t2, t1.c.object_version_id == t2.c.object_version_id))
            .where(and_(t1.c.attribute_name_id.in_(ts_ids),
                        t2.c.attribute_name_id.in_(a_id_ids)))
    )
    result = engine.execute(q)
    lengths = dict()
    for row in result:
        ts_id = row['ts_id']
        a_id_id = row['a_id_id']
        text = row['value']
        length = lengths.get((ts_id, a_id_id))
        if length:
            length['n'] += 1
            length['sum'] += len(text)
            length['sum_sq'] += len(text)**2
        else:
            lengths[(ts_id, a_id_id)] = {'n': 1, 'sum': len(text), 'sum_sq': len(text)**2}
    for c, fv in zip(candidates, feature_values):
        if (c.activity_identifier_attribute_id and not c.relationship_id) and fv['data_type'] == 'string':
            length = lengths.get((c.timestamp_attribute_id, c.activity_identifier_attribute_id))
            if length:
                n = length['n']
                sum_ = length['sum']
                sum_sq = length['sum_sq']
                mean = sum_ / n
                std = sqrt((sum_sq - ((sum_**2) / n)) / (n - 1))
                cv = std / mean
                fv['text_length_mean'] = mean
                fv['text_length_std'] = std
                fv['text_length_cv'] = cv

    # lookup
    ts_ids = {c.timestamp_attribute_id for c in candidates if c.relationship_id}
    a_id_ids = {c.activity_identifier_attribute_id for c in candidates if c.relationship_id}
    rel_ids = {c.relationship_id for c in candidates if c.relationship_id}
    q = (
        select([t1.c.attribute_name_id.label('ts_id'),
                t2.c.attribute_name_id.label('a_id_id'),
                t_rels.c.relationship_id.label('rel_id'),
                t2.c.value])
            .select_from(t1
                         .join(t_rels, t1.c.object_version_id == t_rels.c.source_object_version_id)
                         .join(t2, t_rels.c.target_object_version_id == t2.c.object_version_id))
            .where(and_(t1.c.attribute_name_id.in_(ts_ids),
                        t_rels.c.relationship_id.in_(rel_ids),
                        t2.c.attribute_name_id.in_(a_id_ids)))
    )
    result = engine.execute(q)
    lengths = dict()
    for row in result:
        ts_id = row['ts_id']
        a_id_id = row['a_id_id']
        rel_id = row['rel_id']
        text = row['value']
        length = lengths.get((ts_id, a_id_id, rel_id))
        if length:
            length['n'] += 1
            length['sum'] += len(text)
            length['sum_sq'] += len(text)**2
        else:
            lengths[(ts_id, a_id_id, rel_id)] = {'n': 1, 'sum': len(text), 'sum_sq': len(text)**2}
    for c, fv in zip(candidates, feature_values):
        if c.relationship_id:
            length = lengths.get((c.timestamp_attribute_id, c.activity_identifier_attribute_id, c.relationship_id))
            if length:
                n = length['n']
                sum_ = length['sum']
                sum_sq = length['sum_sq']
                mean = sum_ / n
                std = sqrt((sum_sq - ((sum_**2) / n)) / (n - 1)) if n > 1 else 0
                cv = std / mean if mean > 0 else 0
                fv['text_length_mean'] = mean
                fv['text_length_std'] = std
                fv['text_length_cv'] = cv

    for fv in feature_values:
        if 'text_length_mean' not in fv:
            fv['text_length_mean'] = 0
        if 'text_length_std' not in fv:
            fv['text_length_std'] = 0
        if 'text_length_cv' not in fv:
            fv['text_length_cv'] = 0


def alphabetic_fraction(candidates, feature_values, engine, meta):

    if (_check_already_calculated('alphabetic_fraction', feature_values)
                and _check_already_calculated('alphabetic_fraction', feature_values)):
        return

    def ab_frac(text):
        if len(text) == 0:
            return 0
        ab = [c for c in text if c in (string.ascii_letters + ' ')]
        # spaces are allowed because used to separate words in key phrases. Any other whitspace is not allowed, since
        # it indicates long pieces of text, more than just key phrases
        return len(ab) / len(text)

    for fv in feature_values:
        if 'data_type' not in fv:
            data_type(candidates, feature_values, engine, meta)

    t1 = meta.tables.get('attribute_value').alias()
    t2 = meta.tables.get('attribute_value').alias()
    t_rels = meta.tables.get('relation')

    # in-table
    ts_ids = {c.timestamp_attribute_id for c, fv in zip(candidates, feature_values)
              if (c.activity_identifier_attribute_id and not c.relationship_id)
              and fv['data_type'] == 'string'}
    a_id_ids = {c.activity_identifier_attribute_id for c, fv in zip(candidates, feature_values)
              if (c.activity_identifier_attribute_id and not c.relationship_id)
              and fv['data_type'] == 'string'}
    q = (
        select([t1.c.attribute_name_id.label('ts_id'),
                t2.c.attribute_name_id.label('a_id_id'),
                t2.c.value])
        .select_from(t1.join(t2, t1.c.object_version_id == t2.c.object_version_id))
        .where(and_(t1.c.attribute_name_id.in_(ts_ids),
                    t2.c.attribute_name_id.in_(a_id_ids)))
    )
    result = engine.execute(q)
    fractions = dict()
    for row in result:
        ts_id = row['ts_id']
        a_id_id = row['a_id_id']
        text = row['value']
        if (ts_id, a_id_id) in fractions:
            fractions[(ts_id, a_id_id)]['n'] += 1
            fractions[(ts_id, a_id_id)]['sum'] += ab_frac(text)
        else:
            fractions[(ts_id, a_id_id)] = {'n': 1, 'sum': ab_frac(text)}
    for c, fv in zip(candidates, feature_values):
        if (c.activity_identifier_attribute_id and not c.relationship_id) and fv['data_type'] == 'string':
            fraction = fractions.get((c.timestamp_attribute_id, c.activity_identifier_attribute_id))
            if fraction:
                fv['alphabetic_fraction'] = fraction['sum'] / fraction['n']

    # lookup
    ts_ids = {c.timestamp_attribute_id for c in candidates if c.relationship_id}
    a_id_ids = {c.activity_identifier_attribute_id for c in candidates if c.relationship_id}
    rel_ids = {c.relationship_id for c in candidates if c.relationship_id}
    q = (
        select([t1.c.attribute_name_id.label('ts_id'),
                t2.c.attribute_name_id.label('a_id_id'),
                t_rels.c.relationship_id.label('rel_id'),
                t2.c.value])
        .select_from(t1
                     .join(t_rels, t1.c.object_version_id == t_rels.c.source_object_version_id)
                     .join(t2, t_rels.c.target_object_version_id == t2.c.object_version_id))
        .where(and_(t1.c.attribute_name_id.in_(ts_ids),
                    t_rels.c.relationship_id.in_(rel_ids),
                    t2.c.attribute_name_id.in_(a_id_ids)))
    )
    result = engine.execute(q)
    fractions = dict()
    for row in result:
        ts_id = row['ts_id']
        a_id_id = row['a_id_id']
        rel_id = row['rel_id']
        text = row['value']
        if (ts_id, a_id_id, rel_id) in fractions:
            fractions[(ts_id, a_id_id, rel_id)]['n'] += 1
            fractions[(ts_id, a_id_id, rel_id)]['sum'] += ab_frac(text)
        else:
            fractions[(ts_id, a_id_id, rel_id)] = {'n': 1, 'sum': ab_frac(text)}
    for c, fv in zip(candidates, feature_values):
        if c.relationship_id:
            fraction = fractions.get((c.timestamp_attribute_id, c.activity_identifier_attribute_id, c.relationship_id))
            if fraction:
                fv['alphabetic_fraction'] = fraction['sum'] / fraction['n']

    for fv in feature_values:
        if 'alphabetic_fraction' not in fv:
            fv['alphabetic_fraction'] = 0


def alphabetic_fraction_squared(candidates, feature_values, engine, meta):
    if _check_already_calculated('alphabetic_fraction_squared', feature_values):
        return
    for fv in feature_values:
        if 'alphabetic_fraction' not in fv:
            alphabetic_fraction(candidates, feature_values, engine, meta)
        fv['alphabetic_fraction_squared'] = fv['alphabetic_fraction']**2


all_ = [
    candidate_type,
    data_type,
    timestamp_attribute_id,
    activity_identifier_attribute_id,
    relationship_id,
    nr_timestamps,
    nr_values_where_timestamp,
    nr_unique_values_where_timestamp,
    not_null_ratio,
    log_nr_timestamps,
    log_nr_unique_values_where_timestamp,
    uniqueness_ratio,
    log_uniqueness_ratio,
    nr_no_timestamp,
    timestamp_ratio,
    nr_values_where_no_timestamp,
    no_ts_no_value_ratio,
    text_length_mean,
    log_text_length_mean,
    text_length_std,
    text_length_cv,
    text_length_mean_std_cv,
    alphabetic_fraction,
    alphabetic_fraction_squared,
]


filtered = [
    candidate_type,
    data_type,
    not_null_ratio,
    log_uniqueness_ratio,
    timestamp_ratio,
    no_ts_no_value_ratio,
    log_text_length_mean,
    text_length_cv,
    alphabetic_fraction_squared,
]
