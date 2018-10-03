from sqlalchemy.engine import Engine, ResultProxy, Connection
from sqlalchemy.schema import Table, MetaData, Column
from sqlalchemy.types import STRINGTYPE
from sqlalchemy.sql.expression import select, text, and_,\
    alias, func, distinct, column
from sqlalchemy.orm import scoped_session, sessionmaker, Session
import networkx as nx
from networkx import MultiDiGraph
import copy
from tqdm import tqdm
import json
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import beta
from datetime import datetime
from sqlitedict import SqliteDict
from frozendict import frozendict
import yaml
import pandas as pd
import pickle
import ciso8601


class CaseNotion(dict):

    def __init__(self, d=None):
        if not d:
            dict.__init__(self)
            self['classes_ids'] = set()  # Classes ids
            self['root_id'] = int()  # Root class id
            self['children'] = dict()  # Children map
            self['conv_ids'] = set()  # Converging classes
            self['idc_ids'] = set()  # Identifying classes
            self['relationships'] = set()  # Map of relationships between related classes
        else:
            dict.__init__(self, d)

    def get_classes_ids(self) -> set:
        return self['classes_ids']

    def get_root_id(self) -> int:
        return self['root_id']

    def get_children(self) -> dict:
        return self['children']

    def get_children_of(self, id) -> set:
        if str(id) not in self['children'].keys():
            self['children'][str(id)] = set()
        return self['children'][str(id)]

    def add_child(self, parent, child):
        self.get_children_of(parent).add(child)

    def get_converging_classes(self) -> set:
        return self['conv_ids']

    def get_identifying_classes(self) -> set:
        return self['idc_ids']

    def get_relationships(self) -> set:
        return self['relationships']

    def add_relationship(self, source, target, rs_id, rs_name):
        self.get_relationships().add(
            frozendict({'id': rs_id,
                        'source': source,
                        'target': target,
                        'name': rs_name}))

    def set_root_id(self, id):
        self['root_id'] = id

    def copy(self):
        return copy.deepcopy(self)

    def __hash__(self):
        d = []
        for k in self.keys():
            d.append(str(self[k]).__hash__())
        return str(d).__hash__()


def load_candidates(candidates_path):
    # candidates_mem = yaml.safe_load(open(candidates_path, 'rt'))
    candidates_mem = pickle.load(open(candidates_path, 'rb'))
    return candidates_mem


def save_candidates(candidates, candidates_path):
    candidates_mem = []
    for c in candidates:
        candidates_mem.append(candidates[c])

    # yaml.safe_dump(candidates_mem, open(candidates_path, 'wt'))
    pickle.dump(candidates_mem, open(candidates_path, 'wb'))
    return candidates_mem


def compute_candidates(mm_engine: Engine, min_rel_threshold=0, max_length_path=5, cache_dir: str='.') -> dict:

    candidates = SqliteDict(
        flag='n',
        filename='{}/{}-{}'.format(cache_dir, 'candidates_filecache', datetime.now().timestamp()),
        autocommit=True)
    candidates_aux = SqliteDict(
        flag='n',
        filename='{}/{}-{}'.format(cache_dir, 'candidates_aux_filecache', datetime.now().timestamp()),
        autocommit=True,
        journal_mode="OFF")
    candidates_hash = SqliteDict(
        flag='n',
        filename='{}/{}-{}'.format(cache_dir, 'candidates_hash_filecache', datetime.now().timestamp()),
        autocommit=False,
        journal_mode="OFF")
    # candidates_hash = dict()

    metadata = MetaData(bind=mm_engine)
    metadata.reflect()

    # Get relationships
    relationships = get_relationships(mm_engine)

    # Get all classes
    all_classes = get_all_classes(mm_engine, metadata)

    # Get isolated classes
    isolated = get_isolated_classes(mm_engine, metadata)

    # Get relationships to ignore based on a minimum number of relations present
    rs_stats = get_stats_per_relationship(mm_engine, metadata)
    rs_to_ignore = [rs['rs'] for rs in rs_stats.values() if rs['rr'] <= min_rel_threshold]

    # Create graph with classes as nodes and relationships as edges
    # excluding isolated classes and relationships to ignore

    g = nx.MultiDiGraph()
    g_nodir = nx.MultiGraph()
    g_nomulti = nx.Graph()
    for c in all_classes:
        if c['name'] not in isolated:
            g.add_node(c['id'], **c)
            g_nodir.add_node(c['id'], **c)
            g_nomulti.add_node(c['id'], **c)

    for rs in relationships:
        if rs['rs'] not in rs_to_ignore:
            g.add_edge(rs['source'], rs['target'], key=rs['id'], **rs)
            g_nodir.add_edge(rs['source'], rs['target'], key=rs['id'], **rs)
            g_nomulti.add_edge(rs['source'], rs['target'])

    #import matplotlib.pyplot as plt
    #plt.subplot(121)
    #nx.draw(g, with_labels=True, font_weight='bold')

    # Compute case notions
    # Decompose graph in subgraphs, for each subgraph:
    conn_comp = [g.subgraph(c).copy() for c in nx.weakly_connected_components(g)]

    counter = 0

    comp: MultiDiGraph
    for comp in tqdm(conn_comp, desc='Components'):
        # _ Compute every simple path between pairs of nodes
        for na in tqdm(comp.nodes, desc='Node A'):
            for nb in tqdm(comp.nodes, desc='Node B'):
                ps = [p for p in nx.all_simple_paths(g_nomulti, na, nb, cutoff=max_length_path)]
                for p in tqdm(ps, desc='Simple paths'):
                    pairs = nx.utils.pairwise(p)
                    cand = CaseNotion()
                    cand.get_classes_ids().update(p)
                    cand.get_identifying_classes().update(p)
                    cands = [cand]
                    for pair in pairs:
                        edges = g_nodir.get_edge_data(pair[0], pair[1])
                        if edges.__len__() > 1:
                            extra_cands = list()
                            for e in edges.values():
                                cands_aux = []
                                for c in cands:
                                    cand_aux = c.copy()
                                    cands_aux.append(cand_aux)
                                for c in cands_aux:
                                    c.add_relationship(e['source'], e['target'], e['id'], e['rs'])
                                extra_cands.extend(cands_aux)
                            cands = extra_cands
                        else:
                            for c in cands:
                                for e in edges.values():
                                    c.add_relationship(e['source'], e['target'], e['id'], e['rs'])
                    for cn in cands:
                        for r in cn.get_classes_ids():
                            cn_aux: CaseNotion = cn.copy()
                            cn_aux.set_root_id(r)
                            _build_cn_tree(cn_aux, g)
                            candidates_aux[counter] = cn_aux
                            h = cn_aux.__hash__()
                            lc = candidates_hash.get(h, [])
                            lc.append(counter)
                            candidates_hash[h] = lc
                            counter = counter + 1
                            if counter % 1000 == 0:
                                candidates_aux.sync()
                                candidates_hash.commit(blocking=True)
                                candidates_hash.sync()

        # Add single classes
        for n in tqdm(g.nodes, desc='Single classes'):
            c = CaseNotion()
            c.get_classes_ids().add(n)
            c.get_identifying_classes().add(n)
            c.set_root_id(n)
            candidates_aux[counter] = c
            h = c.__hash__()
            lc = candidates_hash.get(h, [])
            lc.append(counter)
            candidates_hash[h] = lc
            counter = counter + 1
            if counter % 1000 == 0:
                candidates_aux.sync()
                candidates_hash.commit(blocking=True)
                candidates_hash.sync()

    candidates_aux.sync()

    # _ Remove duplicate paths based on set of edges and nodes
    counter_cand = 0
    for lc in tqdm(candidates_hash.values(), total=len(candidates_hash), desc='Removing duplicates'):
        for i, c_i in enumerate(lc):
            c = candidates_aux[c_i]
            unique = True
            for c_a_i in lc[i+1:]:
                c_a = candidates_aux[c_a_i]
                if c_a == c:
                    unique = False
                    break
            if unique:
                candidates[counter_cand] = c
                counter_cand = counter_cand + 1
                if counter_cand % 1000 == 0:
                    candidates.sync()

    candidates.sync()

    return candidates


def _build_cn_tree(case_notion: CaseNotion, g: MultiDiGraph, root=None, nodes: set=None):
    if not nodes:
        nodes = set()
    if not root:
        root = case_notion.get_root_id()
        nodes.add(root)
    children = []
    rss = case_notion.get_relationships()
    for rs in rss:
        if rs['source'] == root:
            child = rs['target']
        elif rs['target'] == root:
            child = rs['source']
        else:
            continue
        if child in case_notion.get_classes_ids():
            if child not in nodes:
                children.append(child)
                nodes.add(child)
                case_notion.add_child(root, child)
    for child in children:
        _build_cn_tree(case_notion, g, child, nodes)


def get_relationships(mm_engine: Engine) -> list:
    relationships = []

    query = select([text('RS.id as id'),
                    text('RS.name as rs'),
                    text('RS.source as source'),
                    text('RS.target as target')]).\
        select_from(text('relationship as RS')).\
        where(text('RS.target != -1'))
    res: ResultProxy = mm_engine.execute(query)

    for r in res:
        rel = {k: r[k] for k in r.keys()}
        relationships.append(rel)

    return relationships


def get_all_classes(mm_engine: Engine, metadata: MetaData = None) -> list:
    classes = []
    if not metadata:
        metadata = MetaData(bind=mm_engine)
        metadata.reflect()
    tb_class: Table = metadata.tables['class']

    query = tb_class.select()
    res = mm_engine.execute(query)

    for r in res:
        classes.append({k: r[k] for k in r.keys()})

    return classes


def build_log_for_case_notion(mm_engine: Engine, case_notion: CaseNotion, proc_name: str,
                              log_name: str, metadata: MetaData = None) -> int:
    if not metadata:
        metadata = MetaData(bind=mm_engine)
        metadata.reflect()

    DBSession: Session = scoped_session(sessionmaker())

    DBSession.remove()
    DBSession.configure(bind=mm_engine, autoflush=False, expire_on_commit=False)

    conn = DBSession.connection()

    tb_proc: Table = metadata.tables['process']
    tb_log: Table = metadata.tables['log']
    tb_log.columns['name'].type = STRINGTYPE
    tb_case: Table = metadata.tables['case']
    tb_ctl: Table = metadata.tables['case_to_log']
    tb_aitc: Table = metadata.tables['activity_instance_to_case']
    tb_atp: Table = metadata.tables['activity_to_process']
    tb_latn: Table = metadata.tables['log_attribute_name']
    tb_latv: Table = metadata.tables['log_attribute_value']
    col_latn_name: Column = tb_latn.columns['name']
    col_latn_name.type = STRINGTYPE
    tb_latv.columns['value'].type = STRINGTYPE
    tb_latv.columns['type'].type = STRINGTYPE

    # Query case identifiers (joined tables of case notion)
    query = _generate_query_from_case_notion(mm_engine, case_notion,
                                             metadata)

    res = conn.execute(query)

    # Create Process
    q = tb_proc.insert().values(name=proc_name)
    proc_id = int(conn.execute(q).lastrowid)

    # Create Log
    q = tb_log.insert().values(process_id=proc_id,
                               name=log_name)
    log_id = conn.execute(q).lastrowid

    latn_cn_id = conn.execute(tb_latn.select().where(col_latn_name == 'case_notion')).first()
    if not latn_cn_id:
        q = tb_latn.insert().values(name='case_notion')
        latn_cn_id = conn.execute(q).lastrowid
    else:
        latn_cn_id = latn_cn_id['id']

    q = tb_latv.insert().values(log_attribute_name_id=latn_cn_id,
                                log_id=log_id,
                                value=yaml.dump(case_notion),
                                type='STRING')
    conn.execute(q)

    case_id_set = set()
    case_id_dict = dict()

    case_ai_map = dict()

    # For each case identifier
    for r in tqdm(res):
        case_id = _build_case_id(r, case_notion)
        if case_id not in case_id_set:
            # Create Case
            q = tb_case.insert().values(name=case_id)
            case_id_db = int(conn.execute(q).lastrowid)
            case_id_set.add(case_id)
            case_id_dict[case_id] = case_id_db
            case_ai_map[case_id_db] = set()
            q = tb_ctl.insert().values(case_id=case_id_db, log_id=log_id)
            conn.execute(q)

        case_id_db = case_id_dict[case_id]

        # For each activity instance of case
        for i in case_notion.get_classes_ids():
            ai_f_name = 'aiid_{}'.format(i)
            if ai_f_name in r.keys():
                # Add AI to Case
                ai_id = r[ai_f_name]
                if ai_id and ai_id not in case_ai_map[case_id_db]:
                    case_ai_map[case_id_db].add(ai_id)
                    q = tb_aitc.insert().values(case_id=case_id_db,
                                                activity_instance_id=ai_id)
                    conn.execute(q)

    act_ids = get_activity_ids_of_log(mm_engine, log_id, conn, metadata)
    # For each activity of cases
    for act_id in act_ids:
        # Add Activity to Process
        q = tb_atp.insert().values(process_id=proc_id, activity_id=act_id)
        conn.execute(q)

    DBSession.commit()

    return log_id


def get_activity_ids_of_log(mm_engine: Engine, log_id: int, conn: Connection, metadata: MetaData=None) -> set:
    if not metadata:
        metadata = MetaData(bind=mm_engine)
        metadata.reflect()

    act_ids = set()

    tb_ai: Table = metadata.tables['activity_instance']
    tb_aitc: Table = metadata.tables['activity_instance_to_case']
    tb_case: Table = metadata.tables['case']
    tb_ctl: Table = metadata.tables['case_to_log']
    col_aitc_ai_id: Column = tb_aitc.columns['activity_instance_id']
    col_aitc_case_id: Column = tb_aitc.columns['case_id']
    col_case_id: Column = tb_case.columns['id']
    col_ctl_case_id: Column = tb_ctl.columns['case_id']
    col_ctl_log_id: Column = tb_ctl.columns['log_id']
    col_ai_id: Column = tb_ai.columns['id']
    col_ai_act_id: Column = tb_ai.columns['activity_id']

    query = select([col_ai_act_id.label('act_id')]).\
        where(and_(col_ai_id == col_aitc_ai_id,
                   col_aitc_case_id == col_case_id,
                   col_case_id == col_ctl_case_id,
                   col_ctl_log_id == log_id))
    res = conn.execute(query)

    for r in res:
        act_id = r['act_id']
        if act_id:
            act_ids.add(act_id)

    return act_ids


def _build_case_id(row, case_notion: CaseNotion) -> str:
    case_id = []

    for c_id in case_notion.get_identifying_classes():
        case_id.append(row['oid_{}'.format(c_id)])

    return str(case_id)


def _generate_query_from_case_notion(mm_engine: Engine, case_notion: CaseNotion,
                                     metadata: MetaData=None):
    tb_obj: Table = metadata.tables['object']
    tb_ov: Table = metadata.tables['object_version']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ev: Table = metadata.tables['event']

    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_ov_id: Column = tb_ov.columns['id']
    col_obj_id: Column = tb_obj.columns['id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']
    col_obj_cl_id: Column = tb_obj.columns['class_id']

    root_c_id = case_notion.get_root_id()

    query_a = alias(select([col_ov_obj_id.label(name='oid_{}'.format(root_c_id)),
                      col_ov_id.label(name='vid_{}'.format(root_c_id)),
                      col_ev_ai_id.label(name='aiid_{}'.format(root_c_id))]).\
        where(and_(col_ov_obj_id == col_obj_id,
                   col_etov_ov_id == col_ov_id,
                   col_etov_ev_id == col_ev_id,
                   col_obj_cl_id == root_c_id)), name='JSQ_{}'.format(root_c_id))

    query = _generate_subquery_cn(mm_engine, case_notion, root_c_id,
                                  query_a, metadata).select()

    return query


def _generate_subquery_cn(mm_engine: Engine, case_notion: CaseNotion,
                          c_id: int, query_in, metadata: MetaData):

    tb_ov: Table = metadata.tables['object_version']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ev: Table = metadata.tables['event']
    tb_rel: Table = metadata.tables['relation']

    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_ov_id: Column = tb_ov.columns['id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']
    col_rel_sov_id: Column = tb_rel.columns['source_object_version_id']
    col_rel_tov_id: Column = tb_rel.columns['target_object_version_id']
    col_rel_rs_id: Column = tb_rel.columns['relationship_id']

    query = query_in

    for child in case_notion.get_children_of(c_id):
        for rs in case_notion.get_relationships():
            rs_id = rs['id']
            rs_source = rs['source']
            rs_target = rs['target']
            if rs_source == c_id and rs_target == child:
                col_rel_stov_A_id = col_rel_sov_id
                col_rel_stov_B_id = col_rel_tov_id
            elif rs_target == c_id and rs_source == child:
                col_rel_stov_A_id = col_rel_tov_id
                col_rel_stov_B_id = col_rel_sov_id
            else:
                continue

            query = query.join(alias(
                select([col_ov_obj_id.label(name='oid_{}'.format(child)),
                        col_ov_id.label(name='vid_{}'.format(child)),
                        col_rel_stov_A_id.label(name='tvid_{}'.format(child)),
                        col_ev_ai_id.label(name='aiid_{}'.format(child))]).\
                where(and_(col_rel_stov_B_id == col_ov_id,
                           col_rel_rs_id == rs_id,
                           col_etov_ov_id == col_ov_id,
                           col_etov_ev_id == col_ev_id)),
                name='JSQ_{}'.format(child)),
                column('JSQ_{0}.tvid_{0}'.format(child), is_literal=True) ==
                column('JSQ_{0}.vid_{0}'.format(c_id), is_literal=True),
                isouter=True)

            query = _generate_subquery_cn(mm_engine, case_notion,
                                          child, query, metadata)

    return query


def get_isolated_classes(mm_engine: Engine, metadata: MetaData=None) -> list:
    isolated = []

    query = select([text('CL.name as cl')]). \
        select_from(text('class as CL')). \
        where(text('CL.id NOT IN (SELECT source as id \
                       FROM relationship \
                       WHERE target != -1 \
                       UNION \
                       SELECT target as id \
                       FROM relationship \
                       WHERE target != -1)'))
    res: ResultProxy = mm_engine.execute(query)

    for r in res:
        cl = {k: r[k] for k in r.keys()}
        isolated.append(cl)

    return isolated


def get_stats_per_relationship(mm_engine: Engine, metadata: MetaData=None) -> dict:
    stats = {}
    if not metadata:
        metadata = MetaData(bind=mm_engine)
        metadata.reflect()
    tb_relationship: Table = metadata.tables['relationship']
    tb_relation: Table = metadata.tables['relation']
    query = select([tb_relationship.columns['name'].label('rs'),
                    func.count(distinct(tb_relation.columns['id'])).label('rr')]).\
        select_from(tb_relationship.join(tb_relation,
                                         tb_relationship.columns['id'] == tb_relation.columns['relationship_id'],
                                         isouter=True)).\
        group_by(tb_relationship.columns['name'])
    res: ResultProxy = mm_engine.execute(query)

    for r in res:
        rel = {k: r[k] for k in r.keys()}
        stats[r['rs']] = rel

    return stats


def get_stats_mm(mm_engine: Engine, metadata: MetaData=None) -> dict:
    stats = {}
    if not metadata:
        metadata = MetaData(bind=mm_engine)
        metadata.reflect()

    tb_class: Table = metadata.tables['class']
    tb_object: Table = metadata.tables['object']
    tb_version: Table = metadata.tables['object_version']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_event: Table = metadata.tables['event']
    tb_ai: Table = metadata.tables['activity_instance']
    query = select([tb_class.columns['id'].label('class'),
                    tb_class.columns['name'].label('class_name'),
                    func.count(distinct(tb_object.columns['id'])).label('o'),
                    func.count(distinct(tb_event.columns['id'])).label('e'),
                    func.count(distinct(tb_ai.columns['activity_id'])).label('act')]).\
        select_from(tb_class.join(tb_object,
                                  tb_object.columns['class_id'] == tb_class.columns['id'],
                                  isouter=True).
                    join(tb_version, tb_version.columns['object_id'] == tb_object.columns['id'],
                         isouter=True).
                    join(tb_etov, tb_etov.columns['object_version_id'] == tb_version.columns['id'],
                         isouter=True).
                    join(tb_event, tb_event.columns['id'] == tb_etov.columns['event_id'],
                         isouter=True).
                    join(tb_ai, tb_ai.columns['id'] == tb_event.columns['activity_instance_id'],
                         isouter=True)).\
        group_by(tb_class.columns['id'])

    res: ResultProxy = mm_engine.execute(query)

    for r in res:
        row = {k: r[k] for k in r.keys()}
        stats[str(r['class'])] = row

    for c_str in stats.keys():
        c = int(c_str)
        stats[c_str]['o_w_ev'] = _num_obj_of_class_with_event(mm_engine, c, metadata)
        stats[c_str]['act_o'] = _sum_unique_act_per_obj_for_class(mm_engine, c, metadata)
        stats[c_str]['min_act_o'] = _min_unique_act_per_obj_for_class(mm_engine, c, metadata)
        stats[c_str]['max_act_o'] = _max_unique_act_per_obj_for_class(mm_engine, c, metadata)
        stats[c_str]['ev_o'] = _sum_ev_per_obj_for_class(mm_engine, c, metadata)
        stats[c_str]['max_ev_o'] = _max_ev_per_obj_for_class(mm_engine, c, metadata)
        stats[c_str]['min_ev_o'] = _min_ev_per_obj_for_class(mm_engine, c, metadata)
        stats[c_str]['act'] = _num_unique_act_per_class(mm_engine, c, metadata)
        stats[c_str]['o'] = _num_obj_of_class(mm_engine, c, metadata)
        stats[c_str]['e'] = _num_ev_per_class(mm_engine, c, metadata)

    return stats


def score_log(wsp, wlod, wae, sp, lod, ae):
    return (wsp * sp) + (wae * ae) + (wlod * lod)


def compute_lod_log(mm_engine: Engine, log_id: int, metadata: MetaData, support: int = None):
    if not support:
        support = compute_support_log(mm_engine, log_id, metadata)

    tb_ctl: Table = metadata.tables['case_to_log']
    col_ctl_case_id: Column = tb_ctl.columns['case_id']
    col_ctl_log_id: Column = tb_ctl.columns['log_id']
    tb_case: Table = metadata.tables['case']
    col_case_id: Column = tb_case.columns['id']
    tb_aitc: Table = metadata.tables['activity_instance_to_case']
    col_aitc_c_id: Column = tb_aitc.columns['case_id']
    col_aitc_ai_id: Column = tb_aitc.columns['activity_instance_id']
    tb_ai: Table = metadata.tables['activity_instance']
    col_ai_id: Column = tb_ai.columns['id']
    col_ai_act_id: Column = tb_ai.columns['activity_id']

    q = select([func.sum(column('A'))]).select_from(
        select([func.count(col_ai_act_id.distinct()).label('A')]).where(and_(col_aitc_ai_id == col_ai_id,
                                                                             col_aitc_c_id == col_case_id,
                                                                             col_case_id == col_ctl_case_id,
                                                                             col_ctl_log_id == log_id)).group_by(
            col_case_id))

    count = mm_engine.execute(q).scalar()

    if support > 0:
        lod = float(count) / float(support)
    else:
        lod = 0.0

    return lod


def compute_ae_log(mm_engine: Engine, log_id: int, metadata: MetaData, support: int = None) -> float:

    if not support:
        support = compute_support_log(mm_engine, log_id, metadata)

    tb_ctl: Table = metadata.tables['case_to_log']
    col_ctl_case_id: Column = tb_ctl.columns['case_id']
    col_ctl_log_id: Column = tb_ctl.columns['log_id']
    tb_case: Table = metadata.tables['case']
    col_case_id: Column = tb_case.columns['id']
    tb_aitc: Table = metadata.tables['activity_instance_to_case']
    col_aitc_c_id: Column = tb_aitc.columns['case_id']
    col_aitc_ai_id: Column = tb_aitc.columns['activity_instance_id']
    tb_ai: Table = metadata.tables['activity_instance']
    col_ai_id: Column = tb_ai.columns['id']
    tb_ev: Table = metadata.tables['event']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']

    q = select([col_ev_id]).where(and_(col_ev_ai_id == col_ai_id,
                                       col_aitc_ai_id == col_ai_id,
                                       col_aitc_c_id == col_case_id,
                                       col_case_id == col_ctl_case_id,
                                       col_ctl_log_id == log_id)).count()

    count = mm_engine.execute(q).scalar()

    if support > 0:
        ae = float(count) / float(support)
    else:
        ae = 0.0

    return ae


def compute_support_log(mm_engine: Engine, log_id: int, metadata: MetaData) -> int:
    tb_ctl: Table = metadata.tables['case_to_log']
    # tb_case: Table = metadata.tables['case']
    # tb_aitc: Table = metadata.tables['activity_instance_to_case']
    # tb_ai: Table = metadata.tables['activity_instance']
    # tb_ev: Table = metadata.tables['event']
    col_ctl_case_id: Column = tb_ctl.columns['case_id']
    col_ctl_log_id: Column = tb_ctl.columns['log_id']
    # col_case_id: Column = tb_case.columns['id']
    # col_aitc_case_id: Column = tb_aitc.columns['case_id']
    # col_aitc_ai_id: Column = tb_aitc.columns['activity_instance_id']
    # col_ai_id: Column = tb_ai.columns['id']
    # col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']

    q = select([col_ctl_case_id]).where(and_(
        col_ctl_log_id == log_id)).distinct().count()

    count = mm_engine.execute(q).scalar()

    return count


def compute_lb_support_cn(mm_engine: Engine, c: CaseNotion, metadata: MetaData, class_stats: dict = None) -> int:
    o_root = _num_obj_of_class_with_event(mm_engine,
                                          c.get_root_id(),
                                          metadata,
                                          class_stats)

    return o_root


def compute_ub_support_cn(mm_engine: Engine, c: CaseNotion, metadata: MetaData, class_stats: dict = None) -> int:
    o_root = _num_obj_of_class_with_event(mm_engine,
                                          c.get_root_id(),
                                          metadata,
                                          class_stats)

    ub_sp = o_root
    for c_id in c.get_classes_ids():
        if c_id != c.get_root_id():
            o_c = _num_obj_of_class(mm_engine, c_id, metadata, class_stats)
            ub_sp = ub_sp * (o_c + 1)

    return ub_sp


def compute_lb_lod_cn(mm_engine: Engine, c: CaseNotion, metadata: MetaData, class_stats: dict = None) -> int:

    sp_ub = compute_ub_support_cn(mm_engine, c, metadata, class_stats)

    o_root = _num_obj_of_class_with_event(mm_engine,
                                          c.get_root_id(),
                                          metadata, class_stats)

    act_o_root = _sum_unique_act_per_obj_for_class(mm_engine, c.get_root_id(),
                                                   metadata, class_stats)

    min_act_o_root = _min_unique_act_per_obj_for_class(mm_engine, c.get_root_id(),
                                                       metadata, class_stats)

    if sp_ub > 0:
        lod_lb = (act_o_root + ((sp_ub - o_root) * min_act_o_root)) / sp_ub
    else:
        lod_lb = 0

    return lod_lb


def compute_ub_lod_cn(mm_engine: Engine, c: CaseNotion, metadata: MetaData, class_stats: dict = None) -> int:

    sum_a = 0
    for c_id in c.get_identifying_classes():
        sum_a = sum_a + _max_unique_act_per_obj_for_class(mm_engine, c_id,
                                                          metadata, class_stats)

    sum_b = 0
    for c_id in c.get_converging_classes():
        sum_b = sum_b + _num_unique_act_per_class(mm_engine, c_id,
                                                  metadata, class_stats)

    lod_ub = sum_a + sum_b

    return lod_ub


def compute_lb_ae_cn(mm_engine: Engine, c: CaseNotion, metadata: MetaData, class_stats: dict = None) -> int:

    sp_ub = compute_ub_support_cn(mm_engine, c, metadata, class_stats)

    o_root = _num_obj_of_class_with_event(mm_engine,
                                          c.get_root_id(),
                                          metadata, class_stats)

    e_o_root = _sum_ev_per_obj_for_class(mm_engine, c.get_root_id(),
                                         metadata, class_stats)

    min_e_o_root = _min_ev_per_obj_for_class(mm_engine, c.get_root_id(),
                                             metadata, class_stats)

    if sp_ub > 0:
        ae_lb = (e_o_root + ((sp_ub - o_root) * min_e_o_root)) / sp_ub
    else:
        ae_lb = 0

    return ae_lb


def compute_ub_ae_cn(mm_engine: Engine, c: CaseNotion, metadata: MetaData, class_stats: dict = None) -> int:

    sum_a = 0
    for c_id in c.get_identifying_classes():
        sum_a = sum_a + _max_ev_per_obj_for_class(mm_engine, c_id,
                                                          metadata, class_stats)

    sum_b = 0
    for c_id in c.get_converging_classes():
        sum_b = sum_b + _num_ev_per_class(mm_engine, c_id,
                                                  metadata, class_stats)

    ae_ub = sum_a + sum_b

    return ae_ub


def _num_obj_of_class_with_event(mm_engine: Engine, class_id: int,
                                 metadata: MetaData, class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['o_w_ev']

    tb_obj: Table = metadata.tables['object']
    tb_ov: Table = metadata.tables['object_version']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ev: Table = metadata.tables['event']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_ov_id: Column = tb_ov.columns['id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']

    q = select([col_obj_id]).where(and_(col_obj_class_id == class_id,
                                        col_ov_obj_id == col_obj_id,
                                        col_etov_ov_id == col_ov_id,
                                        col_ev_id == col_etov_ev_id)). \
        distinct().count()

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _num_obj_of_class(mm_engine: Engine, class_id: int, metadata: MetaData,
                      class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['o']

    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']

    q = select([col_obj_id]).where(col_obj_class_id == class_id).count()

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _num_unique_act_per_class(mm_engine: Engine, class_id: int,
                              metadata: MetaData,
                              class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['act']

    tb_ai: Table = metadata.tables['activity_instance']
    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ai_act_id: Column = tb_ai.columns['activity_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']
    col_ai_id: Column = tb_ai.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.count(col_ai_act_id.distinct())]).where(and_(
        col_ev_ai_id == col_ai_id,
        col_etov_ev_id == col_ev_id,
        col_etov_ov_id == col_ov_id,
        col_ov_obj_id == col_obj_id,
        col_obj_class_id == class_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _num_ev_per_class(mm_engine: Engine, class_id: int,
                              metadata: MetaData,
                              class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['e']

    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.count(col_ev_id.distinct())]).where(and_(
        col_etov_ev_id == col_ev_id,
        col_etov_ov_id == col_ov_id,
        col_ov_obj_id == col_obj_id,
        col_obj_class_id == class_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _min_unique_act_per_obj_for_class(mm_engine: Engine, class_id: int,
                                      metadata: MetaData,
                                      class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['min_act_o']

    tb_ai: Table = metadata.tables['activity_instance']
    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ai_act_id: Column = tb_ai.columns['activity_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']
    col_ai_id: Column = tb_ai.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.min(column('A'))]).select_from(
        select([func.count(col_ai_act_id.distinct()).label('A')]).where(and_(
            col_ev_ai_id == col_ai_id,
            col_etov_ev_id == col_ev_id,
            col_etov_ov_id == col_ov_id,
            col_ov_obj_id == col_obj_id,
            col_obj_class_id == class_id)).group_by(col_obj_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _max_unique_act_per_obj_for_class(mm_engine: Engine, class_id: int,
                                      metadata: MetaData,
                                      class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['max_act_o']

    tb_ai: Table = metadata.tables['activity_instance']
    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ai_act_id: Column = tb_ai.columns['activity_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']
    col_ai_id: Column = tb_ai.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.max(column('A'))]).select_from(
        select([func.count(col_ai_act_id.distinct()).label('A')]).where(and_(
            col_ev_ai_id == col_ai_id,
            col_etov_ev_id == col_ev_id,
            col_etov_ov_id == col_ov_id,
            col_ov_obj_id == col_obj_id,
            col_obj_class_id == class_id)).group_by(col_obj_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _sum_unique_act_per_obj_for_class(mm_engine: Engine, class_id: int,
                                      metadata: MetaData,
                                      class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['act_o']

    tb_ai: Table = metadata.tables['activity_instance']
    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ai_act_id: Column = tb_ai.columns['activity_id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ev_ai_id: Column = tb_ev.columns['activity_instance_id']
    col_ai_id: Column = tb_ai.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.sum(column('A'))]).select_from(
        select([func.count(col_ai_act_id.distinct()).label('A')]).where(and_(
            col_ev_ai_id == col_ai_id,
            col_etov_ev_id == col_ev_id,
            col_etov_ov_id == col_ov_id,
            col_ov_obj_id == col_obj_id,
            col_obj_class_id == class_id)).group_by(col_obj_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _min_ev_per_obj_for_class(mm_engine: Engine, class_id: int,
                              metadata: MetaData,
                              class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['min_ev_o']

    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.min(column('A'))]).select_from(
        select([func.count(col_ev_id.distinct()).label('A')]).where(and_(
            col_etov_ev_id == col_ev_id,
            col_etov_ov_id == col_ov_id,
            col_ov_obj_id == col_obj_id,
            col_obj_class_id == class_id)).group_by(col_obj_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _max_ev_per_obj_for_class(mm_engine: Engine, class_id: int,
                              metadata: MetaData,
                              class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['max_ev_o']

    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.max(column('A'))]).select_from(
        select([func.count(col_ev_id.distinct()).label('A')]).where(and_(
            col_etov_ev_id == col_ev_id,
            col_etov_ov_id == col_ov_id,
            col_ov_obj_id == col_obj_id,
            col_obj_class_id == class_id)).group_by(col_obj_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def _sum_ev_per_obj_for_class(mm_engine: Engine, class_id: int,
                              metadata: MetaData,
                              class_stats: dict = None) -> int:

    if class_stats:
        return class_stats[str(class_id)]['ev_o']

    tb_ev: Table = metadata.tables['event']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_ov: Table = metadata.tables['object_version']
    tb_obj: Table = metadata.tables['object']
    col_obj_class_id: Column = tb_obj.columns['class_id']
    col_obj_id: Column = tb_obj.columns['id']
    col_ev_id: Column = tb_ev.columns['id']
    col_ov_id: Column = tb_ov.columns['id']
    col_ov_obj_id: Column = tb_ov.columns['object_id']
    col_etov_ev_id: Column = tb_etov.columns['event_id']
    col_etov_ov_id: Column = tb_etov.columns['object_version_id']

    q = select([func.sum(column('A'))]).select_from(
        select([func.count(col_ev_id.distinct()).label('A')]).where(and_(
            col_etov_ev_id == col_ev_id,
            col_etov_ov_id == col_ov_id,
            col_ov_obj_id == col_obj_id,
            col_obj_class_id == class_id)).group_by(col_obj_id))

    count = mm_engine.execute(q).scalar()

    if not count:
        count = 0

    return count


def compute_prediction_from_bounds(bounds: dict, w_sp_lb: float, w_lod_lb: float, w_ae_lb: float):
    w_sp_ub = 1 - w_sp_lb
    w_lod_ub = 1 - w_lod_lb
    w_ae_ub = 1 - w_ae_lb

    sp_lb = bounds['sp_lb']
    lod_lb = bounds['lod_lb']
    ae_lb = bounds['ae_lb']
    sp_ub = bounds['sp_ub']
    lod_ub = bounds['lod_ub']
    ae_ub = bounds['ae_ub']

    sp_pred = np.add(np.multiply(sp_lb, w_sp_lb), np.multiply(sp_ub, w_sp_ub)).tolist()
    lod_pred = np.add(np.multiply(lod_lb, w_lod_lb), np.multiply(lod_ub, w_lod_ub)).tolist()
    ae_pred = np.add(np.multiply(ae_lb, w_ae_lb), np.multiply(ae_ub, w_ae_ub)).tolist()

    predictions = {
        'sp': sp_pred,
        'lod': lod_pred,
        'ae': ae_pred,
    }

    return predictions


def compute_ranking(*args, **kwargs) -> list:
    detailed_ranking = compute_detailed_ranking(*args, **kwargs)
    ranking = list(detailed_ranking['cn_id'])
    return ranking


def compute_detailed_ranking(metrics: dict, mode_sp: float, max_sp: int, min_sp: int,
                             mode_lod: float, max_lod: float, min_lod: float,
                             mode_ae: float, max_ae: float, min_ae: float,
                             w_sp: float=0.33, w_lod: float=0.33, w_ae: float=0.33) -> pd.DataFrame:

    sp = metrics['sp']
    lod = metrics['lod']
    ae = metrics['ae']

    sp_a, sp_b, max_sp_glb, min_sp_glb = _estimate_params(
        mode_range=mode_sp,
        max_range=max_sp,
        min_range=min_sp,
        values=sp)

    lod_a, lod_b, max_lod_glb, min_lod_glb = _estimate_params(
        mode_range=mode_lod,
        max_range=max_lod,
        min_range=min_lod,
        values=lod)

    ae_a, ae_b, max_ae_glb, min_ae_glb = _estimate_params(
        mode_range=mode_ae,
        max_range=max_ae,
        min_range=min_ae,
        values=ae)

    sp_scld = scale_minmax(sp, max_sp_glb, min_sp_glb)
    lod_scld = scale_minmax(lod, max_lod_glb, min_lod_glb)
    ae_scld = scale_minmax(ae, max_ae_glb, min_ae_glb)

    max_val_beta_sp = beta.pdf(beta_mode(sp_a, sp_b), sp_a, sp_b)
    max_val_beta_lod = beta.pdf(beta_mode(lod_a, lod_b), lod_a, lod_b)
    max_val_beta_ae = beta.pdf(beta_mode(ae_a, ae_b), ae_a, ae_b)

    sp_score = np.divide(beta.pdf(sp_scld, sp_a, sp_b), max_val_beta_sp)
    lod_score = np.divide(beta.pdf(lod_scld, lod_a, lod_b), max_val_beta_lod)
    ae_score = np.divide(beta.pdf(ae_scld, ae_a, ae_b), max_val_beta_ae)

    weights = [w_sp, w_lod, w_ae]
    scores = np.array([whmean([sp_score_i, lod_score_i, ae_score_i], weights)
              for sp_score_i, lod_score_i, ae_score_i in zip(sp_score, lod_score, ae_score)])

    ranking = np.argsort(-scores).tolist()

    rank_data = {'cn_id': ranking,
                 'score': scores[ranking],
                 'pred_sp': np.array(sp)[ranking],
                 'pred_lod': np.array(lod)[ranking],
                 'pred_ae': np.array(ae)[ranking]
                 }

    detailed_ranking = pd.DataFrame(rank_data)

    return detailed_ranking


def scale_minmax(values: list, max_v: float, min_v: float):

    u = np.subtract(values, min_v)
    b = max_v - min_v

    scld: np.ndarray = np.divide(u, b)
    np.nan_to_num(scld, copy=False)

    return scld


def beta_mode(a, b):

    if a > 1 and b > 1:
        mode = (a - 1) / (a + b - 2)
    elif a == b == 1:
        mode = 0
    elif a == 1 and b > 1:
        mode = 0
    elif b == 1 and a > 1:
        mode = 1
    else:
        raise Exception('Cannot compute mode for (a, b) = ({}, {})'.format(a, b))

    return mode


def compute_bounds_of_candidates(candidates: list, mm_engine: Engine=None,
                                 metadata: MetaData=None,
                                 class_stats: dict = None) -> dict:

    bounds = {'sp_lb': [],
              'sp_ub': [],
              'lod_lb': [],
              'lod_ub': [],
              'ae_lb': [],
              'ae_ub': []}

    for cand in candidates:
        sp_lb = compute_lb_support_cn(mm_engine, cand, metadata, class_stats)
        sp_ub = compute_ub_support_cn(mm_engine, cand, metadata, class_stats)
        lod_lb = compute_lb_lod_cn(mm_engine, cand, metadata, class_stats)
        lod_ub = compute_ub_lod_cn(mm_engine, cand, metadata, class_stats)
        ae_lb = compute_lb_ae_cn(mm_engine, cand, metadata, class_stats)
        ae_ub = compute_ub_ae_cn(mm_engine, cand, metadata, class_stats)

        bounds['sp_lb'].append(sp_lb)
        bounds['sp_ub'].append(sp_ub)
        bounds['lod_lb'].append(lod_lb)
        bounds['lod_ub'].append(lod_ub)
        bounds['ae_lb'].append(ae_lb)
        bounds['ae_ub'].append(ae_ub)

    return bounds


def _estimate_params(mode_range: float, max_range: float, min_range: float, values: list) -> tuple:

    max_value = max(values)
    min_value = min(values)

    if max_range is None:
        max_range = max_value

    if min_range is None:
        min_range = min_value

    if mode_range is None:
        mode_range = min_range + ((max_range - min_range) / 2)

    max_global = max([max_range, max_value])
    min_global = min([min_range, min_value])

    scaled_min = scale_minmax([min_range], max_global, min_global)[0]
    scaled_max = scale_minmax([max_range], max_global, min_global)[0]
    scaled_mode = scale_minmax([mode_range], max_global, min_global)[0]

    a, b = _estimate_a_b(scaled_mode, scaled_min, scaled_max)

    return a, b, max_global, min_global


def _estimate_a_from_b(b: float, mode: float):
    a = (mode*(b-2)+1)/(1-mode)
    return a


def _estimate_b_from_a(a: float, mode: float):
    b = (a*(mode-1)/(-mode))+2-(1/mode)
    return b


def _estimate_a_b(mode: float, min_val: float, max_val: float):

    if (1-mode) > (mode-0):  # Positively skewed: a = 2, b > 2
        if max_val == 0:
            max_val = 0.01
        b = 2 / float(max_val)
        a = _estimate_a_from_b(b, mode)

    elif (1-mode) < (mode-0):  # Negatively skewed: b = 2, a > 2
        if min_val == 1:
            min_val = 0.99
        a = 2 / (1 - float(min_val))
        b = _estimate_b_from_a(a, mode)

    else:  # symmetric: a = b
        a = 2
        b = _estimate_b_from_a(a, mode)
        a = b

    a = max([a, 1])
    b = max([b, 1])

    return a, b


def log_to_dataframe(mm_engine: Engine, mm_meta: MetaData, log_id):

    tb_ctl: Table = mm_meta.tables['case_to_log']
    tb_ai: Table = mm_meta.tables['activity_instance']
    tb_aitc: Table = mm_meta.tables['activity_instance_to_case']
    tb_ev: Table = mm_meta.tables['event']
    tb_act: Table = mm_meta.tables['activity']
    tb_evatn: Table = mm_meta.tables['event_attribute_name']
    tb_evatv: Table = mm_meta.tables['event_attribute_value']

    query = select([tb_ev.c.id, tb_act.c.name, tb_ev.c.lifecycle, tb_ev.c.resource,
                    tb_ev.c.timestamp, tb_aitc.c.case_id, tb_ctl.c.log_id]). \
        where(tb_ctl.c.log_id == log_id).\
        select_from(tb_ev.
                    join(tb_ai, onclause=tb_ev.c.activity_instance_id == tb_ai.c.id).
                    join(tb_act, onclause=tb_ai.c.activity_id == tb_act.c.id).
                    join(tb_aitc, onclause=(tb_ai.c.id == tb_aitc.c.activity_instance_id)).
                    join(tb_ctl, onclause=(tb_ctl.c.case_id == tb_aitc.c.case_id)))

    res: ResultProxy = mm_engine.execute(query)

    data_dict = {}

    for k in res.keys():
        data_dict[k] = []

    for i, r in enumerate(res):
        for k in res.keys():
            if k == 'timestamp':
                ts: datetime = datetime.fromtimestamp(r[k] / 1000.0)
                data_dict[k].append(ts.__str__())
            else:
                data_dict[k].append(r[k])

    df = pd.DataFrame(data=data_dict)

    return df


def list_logs(mm_engine: Engine, mm_meta: MetaData):

    logs = {}

    tb_logs: Table = mm_meta.tables['log']

    query = tb_logs.select()

    res = mm_engine.execute(query)

    for r in res:
        logs[r['id']] = r['name']

    return logs


def log_info(mm_engine: Engine, mm_meta: MetaData, log_id: int):

    info = {'id': log_id,
            'attributes': {}}

    tb_logs: Table = mm_meta.tables['log']
    tb_latn: Table = mm_meta.tables['log_attribute_name']
    tb_latv: Table = mm_meta.tables['log_attribute_value']

    query = select([tb_logs.c.name.label('name'),
                    tb_latn.c.name.label('at_name'),
                    tb_latv.c.value.label('at_v')]).\
        select_from(tb_logs.
                    join(tb_latv, onclause=(tb_latv.c.log_id == tb_logs.c.id)).
                    join(tb_latn, onclause=(tb_latn.c.id == tb_latv.c.log_attribute_name_id))). \
        where(tb_logs.c.id == log_id)

    res = mm_engine.execute(query)

    for r in res:
        info['name'] = r['name']
        info['attributes'][r['at_name']] = r['at_v']

    sp = compute_support_log(mm_engine, log_id, mm_meta)
    info['support'] = sp
    ae = compute_ae_log(mm_engine, log_id, mm_meta, sp)
    info['ae'] = ae
    lod = compute_lod_log(mm_engine, log_id, mm_meta, sp)
    info['lod'] = lod

    if 'case_notion' in info['attributes']:
        cn = yaml.load(info['attributes']['case_notion'])
        info['attributes']['case_notion'] = cn

    return info


def whmean(x, w):
    a = np.sum(w)
    b = np.sum(np.divide(w, x))
    h = a / b
    return h
