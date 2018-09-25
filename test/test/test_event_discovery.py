import eddytools.extraction as ex
import eddytools.events as ev
import os
import shutil
from pprint import pprint
import pickle

from xgboost.sklearn import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

mimic_file_path = 'output/mimic/sample-mimic.slexmm'
mimic_ground_truth = 'data/mimic/ground_truth.json'

train_openslex_file_path = 'data/adw/extracted-adw-mm.slexmm'
dumps_dir = 'output/dumps/adw-ev-disc/'

modified_mm_path = '{}/mm-modified-events.slexmm'.format(dumps_dir)

trained_model = '{}/model_ev_disc.pkl'.format(dumps_dir)
ground_truth_path = 'data/adw/ground_truth.json'


def test_candidates():
    mm_engine = ex.create_mm_engine(train_openslex_file_path)
    mm_meta = ex.get_mm_meta(mm_engine)

    os.makedirs(dumps_dir, exist_ok=True)

    classes = None

    model = ev.train_model(mm_engine=mm_engine, mm_meta=mm_meta,
                           y_true_path=ground_truth_path,
                           classes=classes, model_output=trained_model)

    disc = ev.discover_event_definitions(mm_engine=mm_engine, mm_meta=mm_meta,
                                         classes=classes, dump_dir=dumps_dir,
                                         model=model)

    shutil.copyfile(train_openslex_file_path, modified_mm_path)

    mm_engine_modif = ex.create_mm_engine(modified_mm_path)
    mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

    pred_ts_cand = [c for p, c in zip(disc[ev.CT_TS_FIELD]['predicted'],
                                      disc[ev.CT_TS_FIELD]['candidates']) if p == 1]
    pred_in_table_cand = [c for p, c in zip(disc[ev.CT_IN_TABLE]['predicted'],
                                            disc[ev.CT_IN_TABLE]['candidates']) if p == 1]
    pred_lookup_cand = [c for p, c in zip(disc[ev.CT_LOOKUP]['predicted'],
                                          disc[ev.CT_LOOKUP]['candidates']) if p == 1]

    ev.compute_events(mm_engine_modif, mm_meta_modif, pred_ts_cand[:2])
    ev.compute_events(mm_engine_modif, mm_meta_modif, pred_in_table_cand[:2])
    ev.compute_events(mm_engine_modif, mm_meta_modif, pred_lookup_cand[:2])


def test_candidates_cached():
    mm_engine_train = ex.create_mm_engine(train_openslex_file_path)
    mm_meta_train = ex.get_mm_meta(mm_engine_train)

    cached_dir_train = 'output/adw/ev_disc'

    ts_train_path = '{}/timestamps.json'.format(cached_dir_train)
    candidates_ts_fields_path = '{}/candidates_ts_fields.json'.format(cached_dir_train)
    candidates_in_table_path = '{}/candidates_in_table.json'.format(cached_dir_train)
    candidates_lookup_path = '{}/candidates_lookup.json'.format(cached_dir_train)
    features_in_table_path = '{}/features_in_table.json'.format(cached_dir_train)
    features_lookup_path = '{}/features_lookup.json'.format(cached_dir_train)

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

    predicted_ts_fields = [1 for c in candidates_ts_fields]
    predicted_in_table = aid.predict(X_in_table, candidate_type=ev.CT_IN_TABLE)
    predicted_lookup = aid.predict(X_lookup, candidate_type=ev.CT_LOOKUP)

    shutil.copyfile(train_openslex_file_path, modified_mm_path)

    mm_engine_modif = ex.create_mm_engine(modified_mm_path)
    mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

    pred_ts_cand = [c for p, c in zip(predicted_ts_fields,
                                      candidates_ts_fields) if p == 1]
    pred_in_table_cand = [c for p, c in zip(predicted_in_table,
                                            candidates_in_table) if p == 1]
    pred_lookup_cand = [c for p, c in zip(predicted_lookup,
                                          candidates_lookup) if p == 1]

    ev.compute_events(mm_engine_modif, mm_meta_modif, pred_ts_cand[:2])
    ev.compute_events(mm_engine_modif, mm_meta_modif, pred_in_table_cand[:2])
    ev.compute_events(mm_engine_modif, mm_meta_modif, pred_lookup_cand[:2])


def test_default_model(openslex=train_openslex_file_path, ground_truth=ground_truth_path):
    mm_engine = ex.create_mm_engine(openslex)
    mm_meta = ex.get_mm_meta(mm_engine)
    disc = ev.discover_event_definitions(mm_engine, mm_meta, model='default')
    aid = disc['aid']
    y_true_in_table = aid.load_y_true(disc[ev.CT_IN_TABLE]['candidates'], ground_truth)
    y_true_lookup = aid.load_y_true(disc[ev.CT_LOOKUP]['candidates'], ground_truth)
    y_true_ts_fields = aid.load_y_true(disc[ev.CT_TS_FIELD]['candidates'], ground_truth)
    scores_ts_fields = aid.score(y_true_ts_fields, disc[ev.CT_TS_FIELD]['predicted'])
    scores_in_table = aid.score(y_true_in_table, disc[ev.CT_IN_TABLE]['predicted'])
    scores_lookup = aid.score(y_true_lookup, disc[ev.CT_LOOKUP]['predicted'])
    print('Score Ts Fields')
    pprint(scores_ts_fields)

    print('Score In Table')
    pprint(scores_in_table)

    print('Score Lookup')
    pprint(scores_lookup)


def test_trained_model(openslex_train, ground_truth_train,
                       openslex_test, ground_truth_test):

    mm_engine_train = ex.create_mm_engine(openslex_train)
    mm_meta_train = ex.get_mm_meta(mm_engine_train)
    model_trained = ev.train_model(mm_engine_train, mm_meta_train, y_true_path=ground_truth_train)

    mm_engine_test = ex.create_mm_engine(openslex_test)
    mm_meta_test = ex.get_mm_meta(mm_engine_test)

    disc = ev.discover_event_definitions(mm_engine_test,
                                         mm_meta_test, model=model_trained)

    aid = disc['aid']
    y_true_in_table = aid.load_y_true(disc[ev.CT_IN_TABLE]['candidates'], ground_truth_test)
    y_true_lookup = aid.load_y_true(disc[ev.CT_LOOKUP]['candidates'], ground_truth_test)
    y_true_ts_fields = aid.load_y_true(disc[ev.CT_TS_FIELD]['candidates'], ground_truth_test)
    scores_ts_fields = aid.score(y_true_ts_fields, disc[ev.CT_TS_FIELD]['predicted'])
    scores_in_table = aid.score(y_true_in_table, disc[ev.CT_IN_TABLE]['predicted'])
    scores_lookup = aid.score(y_true_lookup, disc[ev.CT_LOOKUP]['predicted'])
    print('Score Ts Fields')
    pprint(scores_ts_fields)

    print('Score In Table')
    pprint(scores_in_table)

    print('Score Lookup')
    pprint(scores_lookup)


def test_trained_model_cached(openslex_train, ground_truth_train,
                              openslex_test, ground_truth_test,
                              cached_dir_train, cached_dir_test):

    ts_train_path = '{}/timestamps.json'.format(cached_dir_train)
    candidates_ts_fields_path = '{}/candidates_ts_fields.json'.format(cached_dir_train)
    candidates_in_table_path = '{}/candidates_in_table.json'.format(cached_dir_train)
    candidates_lookup_path = '{}/candidates_lookup.json'.format(cached_dir_train)
    features_in_table_path = '{}/features_in_table.json'.format(cached_dir_train)
    features_lookup_path = '{}/features_lookup.json'.format(cached_dir_train)
    model_trained_path = '{}/model.pkl'.format(cached_dir_train)

    ts_test_path = '{}/timestamps.json'.format(cached_dir_test)
    candidates_test_ts_fields_path = '{}/candidates_ts_fields.json'.format(cached_dir_test)
    candidates_test_in_table_path = '{}/candidates_in_table.json'.format(cached_dir_test)
    candidates_test_lookup_path = '{}/candidates_lookup.json'.format(cached_dir_test)
    features_test_in_table_path = '{}/features_in_table.json'.format(cached_dir_test)
    features_test_lookup_path = '{}/features_lookup.json'.format(cached_dir_test)

    if not os.path.exists(cached_dir_test):
        os.mkdir(cached_dir_test)
    if not os.path.exists(cached_dir_train):
        os.mkdir(cached_dir_train)

    mm_engine_train = ex.create_mm_engine(openslex_train)
    mm_meta_train = ex.get_mm_meta(mm_engine_train)

    aid = ev.ActivityIdentifierDiscoverer(engine=mm_engine_train, meta=mm_meta_train,
                                          model=None)

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

    y_true_train_in_table = aid.load_y_true(candidates_in_table, y_true_path=ground_truth_train)
    y_true_train_lookup = aid.load_y_true(candidates_lookup, y_true_path=ground_truth_train)

    class_weight_in_table = compute_class_weight('balanced', [0, 1], y_true_train_in_table)
    class_weight_lookup = compute_class_weight('balanced', [0, 1], y_true_train_lookup)

    classifiers = {
        ev.CT_IN_TABLE: ev.make_sklearn_pipeline(XGBClassifier(max_depth=2, n_estimators=10, random_state=1,
                                                               scale_pos_weight=class_weight_in_table[1])),
        ev.CT_LOOKUP: ev.make_sklearn_pipeline(XGBClassifier(max_depth=2, n_estimators=10, random_state=1,
                                                             scale_pos_weight=class_weight_lookup[1]))}

    aid.set_model(classifiers)

    aid.train_model(X_in_table, y_true_train_in_table, candidate_type=ev.CT_IN_TABLE)
    aid.train_model(X_lookup, y_true_train_lookup, candidate_type=ev.CT_LOOKUP)

    y_pred_ts_fields = [1 for c in candidates_ts_fields]
    y_pred_in_table = aid.predict(X_in_table, candidate_type=ev.CT_IN_TABLE)
    y_pred_lookup = aid.predict(X_lookup, candidate_type=ev.CT_LOOKUP)

    scores_train_in_table = aid.score(y_true_train_in_table, y_pred_in_table)
    scores_train_lookup = aid.score(y_true_train_lookup, y_pred_lookup)

    print('Scores In Table')
    pprint(scores_train_in_table)
    print('Scores Lookup')
    pprint(scores_train_lookup)

    model_trained = classifiers

    with open(model_trained_path, mode='wb') as f:
        pickle.dump(classifiers, f)

    mm_engine_test = ex.create_mm_engine(openslex_test)
    mm_meta_test = ex.get_mm_meta(mm_engine_test)

    aid_test = ev.ActivityIdentifierDiscoverer(engine=mm_engine_test, meta=mm_meta_test,
                                               model=model_trained)

    if os.path.exists(ts_test_path):
        timestamp_attrs_test = aid_test.load_timestamp_attributes(ts_test_path)
    else:
        timestamp_attrs_test = aid_test.get_timestamp_attributes()
        aid_test.save_timestamp_attributes(timestamp_attrs_test, ts_test_path)
    #

    if os.path.exists(candidates_test_ts_fields_path):
        candidates_test_ts_fields = aid_test.load_candidates(candidates_test_ts_fields_path)
    else:
        candidates_test_ts_fields = aid_test.generate_candidates(timestamp_attrs=timestamp_attrs_test,
                                                                 candidate_type=ev.CT_IN_TABLE)
        aid_test.save_candidates(candidates_test_ts_fields, candidates_test_ts_fields_path)

    if os.path.exists(candidates_test_in_table_path):
        candidates_test_in_table = aid_test.load_candidates(candidates_test_in_table_path)
    else:
        candidates_test_in_table = aid_test.generate_candidates(timestamp_attrs=timestamp_attrs_test,
                                                                candidate_type=ev.CT_IN_TABLE)
        aid_test.save_candidates(candidates_test_in_table, candidates_test_in_table_path)

    if os.path.exists(candidates_test_lookup_path):
        candidates_test_lookup = aid_test.load_candidates(candidates_test_lookup_path)
    else:
        candidates_test_lookup = aid_test.generate_candidates(timestamp_attrs=timestamp_attrs_test,
                                                              candidate_type=ev.CT_LOOKUP)
        aid_test.save_candidates(candidates_test_lookup, candidates_test_lookup_path)
    #

    if os.path.exists(features_test_in_table_path):
        feature_values_in_table_test = aid_test.load_features(features_test_in_table_path)
    else:
        feature_values_in_table_test = aid_test.compute_features(candidates_test_in_table, verbose=True)
        aid_test.save_features(feature_values_in_table_test, features_test_in_table_path)

    if os.path.exists(features_test_lookup_path):
        feature_values_lookup_test = aid_test.load_features(features_test_lookup_path)
    else:
        feature_values_lookup_test = aid_test.compute_features(candidates_test_lookup, verbose=True)
        aid_test.save_features(feature_values_lookup_test, features_test_lookup_path)
    #

    pred_test_ts_fields = [1 for c in candidates_test_ts_fields]
    pred_test_in_table = aid_test.predict(feature_values_in_table_test, candidate_type=ev.CT_IN_TABLE)
    pred_test_lookup = aid_test.predict(feature_values_lookup_test, candidate_type=ev.CT_LOOKUP)

    y_true_in_table = aid_test.load_y_true(candidates_test_in_table, ground_truth_test)
    y_true_lookup = aid_test.load_y_true(candidates_test_lookup, ground_truth_test)
    y_true_ts_fields = aid_test.load_y_true(candidates_test_ts_fields, ground_truth_test)
    scores_ts_fields = aid_test.score(y_true_ts_fields, pred_test_ts_fields)
    scores_in_table = aid_test.score(y_true_in_table, pred_test_in_table)
    scores_lookup = aid_test.score(y_true_lookup, pred_test_lookup)
    print('Score Ts Fields')
    pprint(scores_ts_fields)

    print('Score In Table')
    pprint(scores_in_table)

    print('Score Lookup')
    pprint(scores_lookup)


if __name__ == '__main__':
    # print("Test Training")
    # test_candidates()
    #
    print("Test Cached Event Creation")
    test_candidates_cached()
    #
    # print("Test Prediction with Default")
    # test_default_model()

    # if os.path.isfile(mimic_file_path):
        # print("Test Prediction with Default on MIMICIII")
        # test_default_model(openslex=mimic_file_path, ground_truth=mimic_ground_truth)

        # print("Test Prediction with trained model from ADW on MIMICIII (cached)")
        # test_trained_model_cached(openslex_train=train_openslex_file_path, ground_truth_train=ground_truth_path,
        #                           openslex_test=mimic_file_path, ground_truth_test=mimic_ground_truth,
        #                           cached_dir_train='output/adw/ev_disc', cached_dir_test='output/mimic/ev_disc')
