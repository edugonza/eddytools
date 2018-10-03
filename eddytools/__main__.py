"""eddytools

Usage:
  eddytools schema list-schemas <db_url>
  eddytools schema list-classes <db_url> [--details] [--o=OUTPUT_FILE]
  eddytools schema discover <db_url> <output_dir> [--classes=CLASSES_FILE] [--max-fields=K] [--sampling=SAMPLES] [--resume]
  eddytools schema stats <metadata_file>
  eddytools extract <db_url> <output_dir> [<schema_dir>] [--classes=CLASSES_FILE]
  eddytools events <input_db> <output_dir> [--build-events]
  eddytools cases <input_db> <output_dir> [--build-logs --topk=K] [--print_cn=CN_ID --o=OUTPUT_FILE [--show]]
  eddytools logs <input_db> (--list | --info_log=LOG_ID | --export_log=LOG_ID --o=OUTPUT_FILE | --print_cn_log=LOG_ID --o=OUTPUT_FILE [--show])
  eddytools (-h | --help)
  eddytools --version

Options:
  -h --help                 Show this screen.
  --version                 Show version.
  --details                 Show details per class
  --build-events            Build events based on discovered event definitions [default: false].
  --build-logs              Build event logs from case notions [default: false].
  --topk=K                  Show and build only top K case notions and logs
  --info_log=LOG_ID         Show information about the log
  --export_log=LOG_ID       Export the log in csv format in the file specified by --o=OUTPUT_FILE
  --print_cn_log=LOG_ID     Generate a dot file with the case notion used to build log LOG_ID
  --print_cn=CN_ID          Generate a dot file with the case notion with id CN_ID
  --o=OUTPUT_FILE           Output file for stats, a log or a case notion
  --show                    Visualize the case notion graph in a pdf visualizer
  --classes=CLASSES_FILE    File in Json format with a list of class names to extract. If omitted, all will be extracted
  --max-fields=K              Maximum length of keys to discover [default: 4]
  --sampling=SAMPLES        Number of rows per table to sample for schema discovery [default: 0]

"""

import eddytools
import eddytools.extraction as ex
import eddytools.casenotions as cn
import eddytools.events as ev
import eddytools.schema as es
from sqlalchemy.engine import Engine
from sqlalchemy.schema import MetaData
from sqlalchemy import inspect
import os
import shutil
from pathlib import Path
from docopt import docopt
import json
from pprint import pprint
import pandas as pd
from graphviz import Digraph
import pickle


def disc_and_build(mm: Path, new_mm: Path, dump_dir: Path, build_events=False):
    print("Discovering and building events for: {}".format(mm))
    print("In: {}".format(new_mm))
    print("Dumping in: {}".format(dump_dir))

    mm_engine_train = ex.create_mm_engine(mm)
    mm_meta_train = ex.get_mm_meta(mm_engine_train)

    cached_dir_train = dump_dir

    ts_train_path = cached_dir_train / 'timestamps.json'
    candidates_ts_fields_path = cached_dir_train / 'candidates_ts_fields.json'
    candidates_in_table_path = cached_dir_train / 'candidates_in_table.json'
    candidates_lookup_path = cached_dir_train / 'candidates_lookup.json'
    features_in_table_path = cached_dir_train / 'features_in_table.json'
    features_lookup_path = cached_dir_train / 'features_lookup.json'
    final_candidates_ts_fields_path = cached_dir_train / 'final_candidates_ts_fields.json'
    final_candidates_in_table_path = cached_dir_train / 'final_candidates_in_table.json'
    final_candidates_lookup_path = cached_dir_train / 'final_candidates_lookup.json'

    os.makedirs(cached_dir_train, exist_ok=True)

    aid = ev.ActivityIdentifierDiscoverer(engine=mm_engine_train, meta=mm_meta_train,
                                          model='default')

    if os.path.exists(ts_train_path):
        timestamp_attrs = aid.load_timestamp_attributes(ts_train_path)
    else:
        timestamp_attrs = aid.get_timestamp_attributes()
        aid.save_timestamp_attributes(timestamp_attrs, ts_train_path)
    #

    if os.path.exists(candidates_ts_fields_path):
        candidates_ts_fields = aid.load_candidates(candidates_ts_fields_path)
    else:
        candidates_ts_fields = aid.generate_candidates(timestamp_attrs=timestamp_attrs, candidate_type=ev.CT_TS_FIELD)
        aid.save_candidates(candidates_ts_fields, candidates_ts_fields_path)

    if os.path.exists(candidates_in_table_path):
        candidates_in_table = aid.load_candidates(candidates_in_table_path)
    else:
        candidates_in_table = aid.generate_candidates(timestamp_attrs=timestamp_attrs, candidate_type=ev.CT_IN_TABLE)
        aid.save_candidates(candidates_in_table, candidates_in_table_path)

    if os.path.exists(candidates_lookup_path):
        candidates_lookup = aid.load_candidates(candidates_lookup_path)
    else:
        candidates_lookup = aid.generate_candidates(timestamp_attrs=timestamp_attrs, candidate_type=ev.CT_LOOKUP)
        aid.save_candidates(candidates_lookup, candidates_lookup_path)
    #

    if os.path.exists(features_in_table_path):
        X_in_table = aid.load_features(features_in_table_path)
    else:
        X_in_table = aid.compute_features(candidates_in_table, verbose=1)
        aid.save_features(X_in_table, features_in_table_path)

    if os.path.exists(features_lookup_path):
        X_lookup = aid.load_features(features_lookup_path)
    else:
        X_lookup = aid.compute_features(candidates_lookup, verbose=1)
        aid.save_features(X_lookup, features_lookup_path)
    #

    if os.path.exists(final_candidates_ts_fields_path):
        final_candidates_ts_fields = json.load(open(final_candidates_ts_fields_path, 'rt'))
    else:
        predicted_ts_fields = [1 for c in candidates_ts_fields]
        final_candidates_ts_fields = [c for p, c in zip(predicted_ts_fields, candidates_ts_fields) if p == 1]
        json.dump(final_candidates_ts_fields, open(final_candidates_ts_fields_path, 'wt'), indent=True)

    if os.path.exists(final_candidates_in_table_path):
        final_candidates_in_table = json.load(open(final_candidates_in_table_path, 'rt'))
    else:
        predicted_in_table = aid.predict(X_in_table, candidate_type=ev.CT_IN_TABLE)
        final_candidates_in_table = [c for p, c in zip(predicted_in_table, candidates_in_table) if p == 1]
        json.dump(final_candidates_in_table, open(final_candidates_in_table_path, 'wt'), indent=True)

    if os.path.exists(final_candidates_lookup_path):
        final_candidates_lookup = json.load(open(final_candidates_lookup_path, 'rt'))
    else:
        predicted_lookup = aid.predict(X_lookup, candidate_type=ev.CT_LOOKUP)
        final_candidates_lookup = [c for p, c in zip(predicted_lookup, candidates_lookup) if p == 1]
        json.dump(final_candidates_lookup, open(final_candidates_lookup_path, 'wt'), indent=True)

    if build_events:

        shutil.copyfile(mm, new_mm)

        mm_engine_modif = ex.create_mm_engine(new_mm)
        mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

        ev.compute_events(mm_engine_modif, mm_meta_modif, final_candidates_ts_fields)
        ev.compute_events(mm_engine_modif, mm_meta_modif, final_candidates_in_table)
        ev.compute_events(mm_engine_modif, mm_meta_modif, final_candidates_lookup)


def case_notion_candidates_cached(mm_path, dump_dir, build_logs=False, topk=None):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)

    os.makedirs(dump_dir, exist_ok=True)

    class_stats_path = Path(dump_dir, 'class_stats_mm.json')
    candidates_path = Path(dump_dir, 'candidates.pkl')
    bounds_path = Path(dump_dir, 'bounds.json')
    prediction_path = Path(dump_dir, 'prediction.json')
    params_path = Path(dump_dir, 'params.json')
    ranking_path = Path(dump_dir, 'ranking.json')
    mm_modif_path = Path(dump_dir, 'mm-modif-logs.slexmm')

    if os.path.isfile(class_stats_path):
        class_stats = json.load(open('{}/class_stats_mm.json'.format(dump_dir), 'rt'))
    else:
        class_stats = cn.get_stats_mm(mm_engine)
        json.dump(class_stats, open('{}/class_stats_mm.json'.format(dump_dir), 'wt'), indent=True)

    if os.path.isfile(candidates_path):
        candidates_mem = cn.load_candidates(candidates_path)
    else:
        candidates = cn.compute_candidates(mm_engine, cache_dir=dump_dir)
        candidates_mem = cn.save_candidates(candidates, candidates_path)

    if os.path.isfile(bounds_path):
        bounds = json.load(open(bounds_path, 'rt'))
    else:
        bounds = cn.compute_bounds_of_candidates(candidates_mem, mm_engine, mm_meta, class_stats)
        json.dump(bounds, open(bounds_path, 'wt'), indent=True)

    if os.path.isfile(prediction_path):
        pred = json.load(open(prediction_path, 'rt'))
    else:
        pred = cn.compute_prediction_from_bounds(bounds, 0.5, 0.5, 0.5)
        json.dump(pred, open(prediction_path, 'wt'), indent=True)

    if os.path.isfile(params_path):
        params = json.load(open(params_path, 'rt'))
    else:
        params = {
            'mode_sp': None,
            'max_sp': None,
            'min_sp': None,
            'mode_lod': None,
            'max_lod': 10,
            'min_lod': 3,
            'mode_ae': None,
            'max_ae': 3000,
            'min_ae': 0,
            'w_sp': 0.33,
            'w_lod': 0.33,
            'w_ae': 0.33,
        }
        json.dump(params, open(params_path, 'wt'), indent=True)

    pprint(params)

    if os.path.isfile(ranking_path):
        detailed_ranking = pd.DataFrame.from_csv(open(ranking_path, 'rt'))
    else:
        detailed_ranking = cn.compute_detailed_ranking(pred, **params)
        detailed_ranking.to_csv(open(ranking_path, 'wt'))

    ranking = detailed_ranking['cn_id']
    pprint(detailed_ranking)

    if build_logs:

        shutil.copyfile(mm_path, mm_modif_path)

        mm_engine_modif = ex.create_mm_engine(mm_modif_path)
        mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

        if topk:
            ranking_to_build = ranking[:topk]
        else:
            ranking_to_build = ranking

        for idx in ranking_to_build:
            c = candidates_mem[idx]
            proc_name = 'proc_{}'.format(idx)
            log_name = 'log_{}'.format(idx)
            print('Building Log: {}'.format(log_name))
            cn.build_log_for_case_notion(mm_engine_modif, c, proc_name=proc_name,
                                         log_name=log_name, metadata=mm_meta_modif)


def list_logs(mm_path):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)
    logs = cn.list_logs(mm_engine, mm_meta)
    pprint(logs)


def info_log(mm_path, log_id):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)
    info = cn.log_info(mm_engine, mm_meta, log_id)
    pprint(info)


def export_log(mm_path, log_id, output_file):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)

    df: pd.DataFrame = cn.log_to_dataframe(mm_engine=mm_engine, mm_meta=mm_meta, log_id=log_id)
    df.to_csv(output_file, index_label='idx')


def print_cn_log(mm_path, log_id, output_file, view):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)
    info = cn.log_info(mm_engine, mm_meta, log_id)
    if 'case_notion' in info['attributes']:
        dump_cn_dot(mm_engine, mm_meta, info['attributes']['case_notion'], output_file, view)
    else:
        raise Exception('No case notion to show')


def print_cn(mm_path, cn_id, cn_dir, output_file, view):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)
    try:
        candidates = pickle.load(open(Path(cn_dir, 'candidates.pkl'), 'rb'))
        case_notion = candidates[cn_id]
        dump_cn_dot(mm_engine, mm_meta, case_notion, output_file, view)
    except:
        raise Exception('No case notion to show')


def dump_cn_dot(mm_engine, mm_meta, case_notion: cn.CaseNotion, output_file, view):

    classes = cn.get_all_classes(mm_engine, mm_meta)

    dcl = {}
    for c in classes:
        dcl[int(c['id'])] = str(c['name'])

    root_id = case_notion.get_root_id()
    root_name = dcl[root_id]

    graph = Digraph(comment='Case notion',
                    graph_attr={'splines': 'true',
                                'overlap': 'false',
                                'esep': '3',
                                'root': str(root_id)})

    graph.node(str(root_id), label=root_name, shape='box', style="setlinewidth(3)")
    for c in case_notion.get_classes_ids():
        if not c == root_id:
            if c in case_notion.get_converging_classes():
                graph.node(str(c), label=dcl[c], shape='box', peripheries="2")
            else:
                graph.node(str(c), label=dcl[c], shape='box')

    for c_p in case_notion.get_children():
        for c_s in case_notion.get_children()[c_p]:
            graph.edge(str(c_p), str(c_s))

    for rs in case_notion.get_relationships():
        source = str(rs['source'])
        target = str(rs['target'])
        label = ''  # str(rs['name'])
        graph.edge(source, target, label=label, style='dashed')

    graph.save(filename=output_file)
    graph.render(view=view)


def discover_schema(db_url, output_dir, classes_file, max_fields_key=4, resume=False, sampling=0):
    db_engine: Engine = ex.create_db_engine_from_url(db_url)
    if classes_file:
        classes = json.load(open(classes_file, 'rt'))
    else:
        classes = None
    es.full_discovery_from_engine(db_engine, dump_dir=output_dir, classes=classes,
                                  max_fields_key=max_fields_key,
                                  resume=resume, sampling=sampling)


def extract_data(db_url, output_dir, schema_dir=None, classes_file=None):
    db_engine: Engine = ex.create_db_engine_from_url(db_url)
    if schema_dir:
        metadata = pickle.load(open(Path(schema_dir,'metadata_filtered.pickle'), mode='rb'))

        discovered_pks = json.load(open(Path(schema_dir,'pruned_pks.json'), mode='rt'))
        discovered_fks = json.load(open(Path(schema_dir,'filtered_fks.json'), mode='rt'))

        db_meta = es.create_custom_metadata(db_engine, None,
                                            discovered_pks, discovered_fks, metadata=metadata)
    else:
        db_meta: MetaData = ex.get_metadata(db_engine)

    if classes_file:
        classes = json.load(open(classes_file, 'rt'))
    else:
        classes = None

    db_engine.dispose()
    ex.extraction_from_db(Path(output_dir, 'mm-extracted.slexmm'), output_dir,
                          db_engine, overwrite=True, classes=classes,
                          metadata=db_meta)


def schema_list_schemas(db_url):
    db_engine: Engine = ex.create_db_engine_from_url(db_url)
    insp = inspect(db_engine)
    schemas = insp.get_schema_names()
    schemas_json = json.dumps(schemas, indent=True)
    print(schemas_json)


def schema_list_classes(db_url, details=False, output_file=False):
    db_engine: Engine = ex.create_db_engine_from_url(db_url)
    metadata: MetaData = ex.get_metadata(db_engine)
    classes = es.retrieve_classes(metadata)
    if details:
        classes_details = []
        for cl in classes:
            rows = es.count_rows(db_engine, metadata, cl)
            cl_det = {'cl': cl,
                      'rows': rows}
            classes_details.append(cl_det)
            print(cl_det)
        df = pd.DataFrame(classes_details)
        if output_file:
            df.to_csv(output_file)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df)
    else:
        classes_json = json.dumps(classes, indent=True)
        if output_file:
            json.dump(classes, open(output_file, 'wt'), indent=True)
        print(classes_json)


def print_schema_stats(meta_path):
    meta: MetaData = pickle.load(open(meta_path, mode='rb'))
    stats = es.schema_stats(meta)

    print('# of tables: {}'.format(stats['n_tables']))
    print('# of columns: {}'.format(stats['n_columns']))
    print('# of timestamp columns: {}'.format(stats['n_ts_cols']))

    print('Mean cols per table: {}'.format(stats['avg_cols_p_table']))
    print('Median cols per table: {}'.format(stats['median_cols_p_table']))
    print('Mean timestamp cols per table: {}'.format(stats['avg_ts_cols_p_table']))
    print('Median timestamp cols per table: {}'.format(stats['mediam_ts_cols_p_table']))

    print('# of pks: {}'.format(stats['n_pks']))
    print('# of uks: {}'.format(stats['n_uks']))
    print('# of fks: {}'.format(stats['n_fks']))

    print('Mean uks per table: {}'.format(stats['avg_uks_p_table']))
    print('Median uks per table: {}'.format(stats['median_uks_p_table']))
    print('Mean fks per table: {}'.format(stats['avg_fks_p_table']))
    print('Median fks per table: {}'.format(stats['median_fks_p_table']))


if __name__ == '__main__':

    arguments = docopt(__doc__, version=eddytools.__version__)

    if arguments['schema']:
        db_url = arguments['<db_url>']
        if arguments['list-schemas']:
            schema_list_schemas(db_url)
        elif arguments['list-classes']:
            details = arguments['--details']
            output_file = arguments['--o']
            schema_list_classes(db_url, details, output_file)
        elif arguments['discover']:
            output_dir = arguments['<output_dir>']
            classes_file = arguments['--classes']
            max_fields = int(arguments['--max-fields'])
            resume = arguments['--resume']
            sampling = int(arguments['--sampling'])
            discover_schema(db_url, output_dir, classes_file, max_fields_key=max_fields,
                            resume=resume, sampling=sampling)
        elif arguments['stats']:
            metadata_file = arguments['<metadata_file>']
            print_schema_stats(metadata_file)
    elif arguments['extract']:
        db_url = arguments['<db_url>']
        schema_dir = arguments['<schema_dir>']
        output_dir = arguments['<output_dir>']
        classes_file = arguments['--classes']
        extract_data(db_url, output_dir, schema_dir, classes_file)
    elif arguments['events']:
        input_mm = Path(arguments['<input_db>'])
        output_dir = Path(arguments['<output_dir>'])
        output_mm = Path(output_dir, 'mm-modif.slexmm')
        build_events = arguments['--build-events']
        disc_and_build(mm=input_mm, new_mm=output_mm, dump_dir=output_dir, build_events=build_events)

    elif arguments['cases']:
        input_mm = Path(arguments['<input_db>'])
        output_dir = Path(arguments['<output_dir>'])
        build_logs = arguments['--build-logs']
        if arguments['--topk']:
            topk = int(arguments['--topk'])
        else:
            topk = None
        case_notion_candidates_cached(mm_path=input_mm, dump_dir=output_dir, build_logs=build_logs, topk=topk)
        if arguments['--print_cn']:
            cn_id = int(arguments['--print_cn'])
            output_file = Path(arguments['--o'])
            show = arguments['--show']
            print_cn(input_mm, cn_id, output_dir, output_file, show)
    elif arguments['logs']:
        input_mm = Path(arguments['<input_db>'])
        if arguments['--list']:
            list_logs(input_mm)
        elif arguments['--info_log']:
            log_id = arguments['--info_log']
            info_log(input_mm, log_id)
        elif arguments['--print_cn_log']:
            log_id = arguments['--print_cn_log']
            output_file = Path(arguments['--o'])
            show = arguments['--show']
            print_cn_log(input_mm, log_id, output_file, show)
        elif arguments['--export_log']:
            log_id = arguments['--export_log']
            output_file = Path(arguments['--o'])
            export_log(input_mm, log_id, output_file)
