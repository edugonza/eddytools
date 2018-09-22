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

modified_mm = '{}/mm-modified-events.slexmm'.format(dumps_dir)

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

    predicted, candidates, aid = ev.discover_event_definitions(mm_engine=mm_engine, mm_meta=mm_meta,
                                               classes=classes, dump_dir=dumps_dir,
                                               model=model)

    shutil.copyfile(train_openslex_file_path, modified_mm)

    mm_engine_modif = ex.create_mm_engine(modified_mm)
    mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

    ev.compute_events(mm_engine_modif, mm_meta_modif, [c for p, c in zip(predicted, candidates) if p == 1])


def test_candidates_cached():
    mm_engine_train = ex.create_mm_engine(train_openslex_file_path)
    mm_meta_train = ex.get_mm_meta(mm_engine_train)

    cached_dir_train = 'output/adw/ev_disc'

    ts_train_path = '{}/timestamps.json'.format(cached_dir_train)
    candidates_train_path = '{}/candidates.json'.format(cached_dir_train)
    features_train_path = '{}/features.json'.format(cached_dir_train)

    os.makedirs(dumps_dir, exist_ok=True)

    aid = ev.ActivityIdentifierDiscoverer(engine=mm_engine_train, meta=mm_meta_train,
                                          model='default')

    if os.path.exists(ts_train_path):
        timestamp_attrs = aid.load_timestamp_attributes(ts_train_path)
    else:
        timestamp_attrs = aid.get_timestamp_attributes()
        aid.save_timestamp_attributes(timestamp_attrs, ts_train_path)
    #

    if os.path.exists(candidates_train_path):
        candidates = aid.load_candidates(candidates_train_path)
    else:
        candidates = aid.generate_candidates(timestamp_attrs=timestamp_attrs, candidate_types='in_table')
        aid.save_candidates(candidates, candidates_train_path)
    #

    if os.path.exists(features_train_path):
        X = aid.load_features(features_train_path)
    else:
        X = aid.compute_features(candidates, verbose=1)
        aid.save_features(X, features_train_path)
    #

    predicted = aid.predict(X)

    shutil.copyfile(train_openslex_file_path, modified_mm)

    mm_engine_modif = ex.create_mm_engine(modified_mm)
    mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

    ev.compute_events(mm_engine_modif, mm_meta_modif, [c for p, c in zip(predicted, candidates) if p == 1])


def test_default_model(openslex=train_openslex_file_path, ground_truth=ground_truth_path):
    mm_engine = ex.create_mm_engine(openslex)
    mm_meta = ex.get_mm_meta(mm_engine)
    pred, candidates, aid = ev.discover_event_definitions(mm_engine, mm_meta, model='default')
    y_true = aid.load_y_true(candidates, ground_truth)
    scores = aid.score(y_true, pred)
    pprint(scores)


def test_trained_model(openslex_train, ground_truth_train,
                       openslex_test, ground_truth_test):

    mm_engine_train = ex.create_mm_engine(openslex_train)
    mm_meta_train = ex.get_mm_meta(mm_engine_train)
    model_trained = ev.train_model(mm_engine_train, mm_meta_train, y_true_path=ground_truth_train)

    mm_engine_test = ex.create_mm_engine(openslex_test)
    mm_meta_test = ex.get_mm_meta(mm_engine_test)

    pred_test, candidates_test, aid_test = ev.discover_event_definitions(mm_engine_test,
                                                                         mm_meta_test, model=model_trained)

    y_true_test = aid_test.load_y_true(candidates_test, ground_truth_test)
    scores_test = aid_test.score(y_true_test, pred_test)
    pprint(scores_test)


def test_trained_model_cached(openslex_train, ground_truth_train,
                              openslex_test, ground_truth_test,
                              cached_dir_train, cached_dir_test):

    ts_train_path = '{}/timestamps.json'.format(cached_dir_train)
    candidates_train_path = '{}/candidates.json'.format(cached_dir_train)
    features_train_path = '{}/features.json'.format(cached_dir_train)
    model_trained_path = '{}/model.pkl'.format(cached_dir_train)

    ts_test_path = '{}/timestamps.json'.format(cached_dir_test)
    candidates_test_path = '{}/candidates.json'.format(cached_dir_test)
    features_test_path = '{}/features.json'.format(cached_dir_test)

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

    if os.path.exists(candidates_train_path):
        candidates = aid.load_candidates(candidates_train_path)
    else:
        candidates = aid.generate_candidates(timestamp_attrs=timestamp_attrs, candidate_types='in_table')
        aid.save_candidates(candidates, candidates_train_path)
    #

    if os.path.exists(features_train_path):
        X = aid.load_features(features_train_path)
    else:
        X = aid.compute_features(candidates, verbose=1)
        aid.save_features(X, features_train_path)
    #

    y_true = aid.load_y_true(candidates, y_true_path=ground_truth_train)

    class_weight = compute_class_weight('balanced', [0, 1], y_true)
    classifier = ev.make_sklearn_pipeline(XGBClassifier(max_depth=2, n_estimators=10, random_state=1,
                                                        scale_pos_weight=class_weight[1]))

    aid.set_model(classifier)

    aid.train_model(X, y_true)

    y_pred = aid.predict(X)

    scores = aid.score(y_true, y_pred)

    pprint(scores)

    model_trained = classifier

    with open(model_trained_path, mode='wb') as f:
        pickle.dump(classifier, f)

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

    if os.path.exists(candidates_test_path):
        candidates_test = aid_test.load_candidates(candidates_test_path)
    else:
        candidates_test = aid_test.generate_candidates(timestamp_attrs=timestamp_attrs_test, candidate_types='in_table')
        aid_test.save_candidates(candidates_test, candidates_test_path)
    #

    if os.path.exists(features_test_path):
        feature_values_test = aid_test.load_features(features_test_path)
    else:
        feature_values_test = aid_test.compute_features(candidates_test, verbose=True)
        aid_test.save_features(feature_values_test, features_test_path)

    pred_test = aid_test.predict(feature_values_test)

    y_true_test = aid_test.load_y_true(candidates_test, ground_truth_test)
    scores_test = aid_test.score(y_true_test, pred_test)
    pprint(scores_test)


if __name__ == '__main__':
    print("Test Training")
    test_candidates()

    print("Test Cached Event Creation")
    test_candidates_cached()

    print("Test Prediction with Default")
    test_default_model()

    if os.path.isfile(mimic_file_path):
        print("Test Prediction with Default on MIMICIII")
        test_default_model(openslex=mimic_file_path, ground_truth=mimic_ground_truth)

        print("Test Prediction with trained model from ADW on MIMICIII (cached)")
        test_trained_model_cached(openslex_train=train_openslex_file_path, ground_truth_train=ground_truth_path,
                                  openslex_test=mimic_file_path, ground_truth_test=mimic_ground_truth,
                                  cached_dir_train='output/adw/ev_disc', cached_dir_test='output/mimic/ev_disc')
