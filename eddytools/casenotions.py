from sqlalchemy.engine import Engine, ResultProxy
from sqlalchemy.schema import Table, MetaData
from sqlalchemy.sql.expression import select, text, and_, table, func, distinct
import networkx as nx


def compute_candidates(mm_engine: Engine, min_rel_threshold=0) -> dict:
    candidates = {}

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
            g.add_edge(rs['source'], rs['target'], key=rs['rs'])

    #import matplotlib.pyplot as plt
    #plt.subplot(121)
    #nx.draw(g, with_labels=True, font_weight='bold')

    # Compute case notions
    # Decompose graph in subgraphs, for each subgraph:

    conn_comp = nx.weakly_connected_component_subgraphs(g)

    for comp in conn_comp:
        # _ Compute every simple path between pairs of nodes
        # _ Compute communities
        # _ Remove duplicate paths based on set of edges
        pass


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
