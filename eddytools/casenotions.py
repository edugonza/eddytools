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


class CaseNotion(dict):

    def __init__(self):
        dict.__init__(self)
        self['classes_ids'] = list()  # Classes ids
        self['root_id'] = int()  # Root class id
        self['children'] = dict()  # Children map
        self['conv_ids'] = list()  # Converging classes
        self['idc_ids'] = list()  # Identifying classes
        self['relationships'] = list()  # Map of relationships between related classes

    def get_classes_ids(self) -> list:
        return self['classes_ids']

    def get_root_id(self) -> int:
        return self['root_id']

    def get_children(self) -> dict:
        return self['children']

    def get_children_of(self, id) -> id:
        if id not in self['children'].keys():
            self['children'][id] = []
        return self['children'][id]

    def add_child(self, parent, child):
        self.get_children_of(parent).append(child)

    def get_converging_classes(self) -> list:
        return self['conv_ids']

    def get_identifying_classes(self) -> list:
        return self['idc_ids']

    def get_relationships(self) -> list:
        return self['relationships']

    def add_relationship(self, source, target, rs_id, rs_name):
        self.get_relationships().append(
            {'id': rs_id,
             'source': source,
             'target': target,
             'name': rs_name})

    def set_root_id(self, id):
        self['root_id'] = id

    def copy(self):
        return copy.deepcopy(self)


def compute_candidates(mm_engine: Engine, min_rel_threshold=0) -> list:
    candidates = list()

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
    for c in all_classes:
        if c['name'] not in isolated:
            g.add_node(c['id'], **c)
            g_nodir.add_node(c['id'], **c)

    for rs in relationships:
        if rs['rs'] not in rs_to_ignore:
            g.add_edge(rs['source'], rs['target'], key=rs['id'], **rs)
            g_nodir.add_edge(rs['source'], rs['target'], key=rs['id'], **rs)

    #import matplotlib.pyplot as plt
    #plt.subplot(121)
    #nx.draw(g, with_labels=True, font_weight='bold')

    # Compute case notions
    # Decompose graph in subgraphs, for each subgraph:
    conn_comp = nx.weakly_connected_component_subgraphs(g)

    candidates_aux = list()

    comp: MultiDiGraph
    for comp in conn_comp:
        # _ Compute every simple path between pairs of nodes
        for na in comp.nodes:
            for nb in comp.nodes:
                ps = nx.all_simple_paths(g_nodir, na, nb)
                for p in ps:
                    pairs = nx.utils.pairwise(p)
                    cand = CaseNotion()
                    cand.get_classes_ids().extend(p)
                    cand.get_identifying_classes().extend(p)
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
                            candidates_aux.append(cn_aux)

        # Add single classes
        for n in g.nodes:
            c = CaseNotion()
            c.get_classes_ids().append(n)
            c.set_root_id(n)
            candidates_aux.append(c)

        # _ Remove duplicate paths based on set of edges and nodes
        for c in candidates_aux:
            unique = True
            for c_a in candidates:
                if c_a == c:
                    unique = False
                    break
            if unique:
                candidates.append(c)

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
                                value=json.dumps(case_notion),
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
                   col_etov_ov_id == col_obj_id,
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

    # SELECT  CL.name as class,
    #         count(distinct OBJ.id) as o,
    #         count(distinct EV.id) as e,
    #         count(distinct AI.activity_id) as act
    # FROM
    #         class as CL
    # LEFT OUTER JOIN object as OBJ ON CL.id = OBJ.class_id
    # LEFT OUTER JOIN object_version as OV ON OBJ.id = OV.object_id
    # LEFT OUTER JOIN event_to_object_version as ETOV ON OV.id = ETOV.object_version_id
    # LEFT OUTER JOIN event as EV ON ETOV.event_id = EV.id
    # LEFT OUTER JOIN activity_instance as AI ON AI.id = EV.activity_instance_id
    #
    # GROUP BY CL.name

    tb_class: Table = metadata.tables['class']
    tb_object: Table = metadata.tables['object']
    tb_version: Table = metadata.tables['object_version']
    tb_etov: Table = metadata.tables['event_to_object_version']
    tb_event: Table = metadata.tables['event']
    tb_ai: Table = metadata.tables['activity_instance']
    query = select([tb_class.columns['name'].label('class'),
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
        group_by(tb_class.columns['name'])

    res: ResultProxy = mm_engine.execute(query)

    for r in res:
        row = {k: r[k] for k in r.keys()}
        stats[r['class']] = row

    return stats