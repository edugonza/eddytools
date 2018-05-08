from sqlalchemy.engine import Engine, ResultProxy
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.sql.expression import select, text, and_, table, func, distinct
import networkx as nx
from networkx import MultiDiGraph


def compute_candidates(mm_engine: Engine, min_rel_threshold=0) -> list:
    candidates = list()

    metadata = MetaData(bind=mm_engine)
    metadata.reflect()

    # Get relationships
    relationships = get_relationships(mm_engine, metadata)

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
    for c in all_classes:
        if c['name'] not in isolated:
            g.add_node(c['name'])

    for rs in relationships:
        if rs['rs'] not in rs_to_ignore:
            g.add_edge(rs['source'], rs['target'], key=rs['rs'], name=rs['rs'])

    g.add_edge('BOOKING', 'CUSTOMER', key='TEST_LINK', name='TEST_LINK')

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
                ps = nx.all_simple_paths(comp, na, nb)
                for p in ps:
                    pairs = nx.utils.pairwise(p)
                    cand = dict()
                    cand['nodes'] = p
                    cand['edges'] = []
                    cands = [cand]
                    for pair in pairs:
                        edges = comp.get_edge_data(pair[0], pair[1])
                        if edges.__len__() > 1:
                            extra_cands = list()
                            for e in edges.values():
                                cands_aux = []
                                for c in cands:
                                    cand_aux = {'nodes': c['nodes'].copy(),
                                                'edges': c['edges'].copy(),
                                                }
                                    cands_aux.append(cand_aux)
                                for c in cands_aux:
                                    c['edges'].append(e['name'])
                                extra_cands.extend(cands_aux)
                            cands = extra_cands
                        else:
                            for c in cands:
                                for e in edges.values():
                                    c['edges'].append(e['name'])
                    candidates_aux.extend(cands)

        # Add single classes
        for n in g.nodes:
            candidates_aux.append({'nodes': n, 'edges': []})

        # _ Remove duplicate paths based on set of edges and nodes
        for c in candidates_aux:
            unique = True
            for c_a in candidates:
                if c_a['nodes'] == c['nodes'] and c_a['edges'] == c['edges']:
                    unique = False
                    break
            if unique:
                candidates.append(c)

    return candidates


def get_relationships(mm_engine: Engine, metadata: MetaData = None) -> list:
    relationships = []

    query = select([text('RS.name as rs'),
                    text('CLS.name as source'),
                    text('CLT.name as target')]).\
        select_from(text('class as CLS')).\
        select_from(text('class as CLT')). \
        select_from(text('relationship as RS')).\
        where(and_(text('RS.source = CLS.id'),
                   text('RS.target = CLT.id'),
                   text('RS.target != -1')))
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