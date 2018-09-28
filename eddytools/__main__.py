"""eddytools

Usage:
  eddytools events <input_db> <output_dir> [--build-events]
  eddytools cases <input_db> <output_dir> [--build-logs --topk=K]
  eddytools logs <input_db> (--list | --log_id <log_id> <output_file>)
  eddytools (-h | --help)
  eddytools --version

Options:
  -h --help        Show this screen.
  --version        Show version.
  --build-events   Build events based on discovered event definitions [default: false].
  --build-logs     Build event logs from case notions [default: false].
  --topk=K         Show and build only top K case notions and logs

"""

import eddytools
import eddytools.extraction as ex
import eddytools.casenotions as cn
import eddytools.events as ev
import os
import shutil
from pathlib import Path
from docopt import docopt
import json
from pprint import pprint
import pandas as pd


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
        ranking = json.load(open(ranking_path, 'rt'))
    else:
        ranking = cn.compute_ranking(pred, **params)
        json.dump(ranking, open(ranking_path, 'wt'), indent=True)

    pprint(ranking)

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


def export_log(mm_path, log_id, output_file):
    mm_engine = ex.create_mm_engine(mm_path)
    mm_meta = ex.get_mm_meta(mm_engine)

    df: pd.DataFrame = cn.log_to_dataframe(mm_engine=mm_engine, mm_meta=mm_meta, log_id=log_id)
    df.to_csv(output_file, index_label='idx')


if __name__ == '__main__':

    arguments = docopt(__doc__, version=eddytools.__version__)

    if arguments['events']:
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

    elif arguments['logs']:
        input_mm = Path(arguments['<input_db>'])
        if arguments['--list']:
            list_logs(input_mm)
        elif arguments['--log_id']:
            log_id = arguments['<log_id>']
            output_file = Path(arguments['<output_file>'])
            export_log(input_mm, log_id, output_file)
