import eddytools.extraction as ex
import eddytools.events as ev
import json
import os
import pickle
import shutil
from eddytools.events import Candidate


#openslex_file_path = 'data/mm/ds2-sample.slexmm'
#openslex_file_path = 'output/mimic/sample-mimic.slexmm'
#y_true_path = 'data/mimic/ground_truth.json'
#dumps_dir = 'output/dumps/mimic-ev-disc/'

openslex_file_path = 'output/adw/mm.slexmm'
dumps_dir = 'output/dumps/adw-ev-disc/'

candidates_path = '{}/candidates.json'.format(dumps_dir)
features_path = '{}/features_filtered.json'.format(dumps_dir)

modified_mm = '{}/mm-modified-events.slexmm'.format(dumps_dir)


def test_candidates():
    mm_engine = ex.create_mm_engine(openslex_file_path)
    mm_meta = ex.get_mm_meta(mm_engine)

    os.makedirs(dumps_dir, exist_ok=True)

    classes = None

    # trained_model, encoders = ev.train_classifier(mm_engine, mm_meta, y_true_path, classes,
    #                                     dump_dir=dumps_dir)

    # trained_model, encoders = ev.train_classifier_cached(mm_engine, mm_meta, candidates_path,
    #                                                      y_true_path, features_path,
    #                                                      dump_dir=dumps_dir)

    # pickle.dump(trained_model, open('{}/ev_disc_model.pkl'.format(dumps_dir), mode='wb'))
    # pickle.dump(encoders, open('{}/ev_disc_encoders.pkl'.format(dumps_dir), mode='wb'))

    trained_model = pickle.load(open('{}/ev_disc_model.pkl'.format(dumps_dir), mode='rb'))
    encoders = pickle.load(open('{}/ev_disc_encoders.pkl'.format(dumps_dir), mode='rb'))

    # candidates, _ = ev.discover_event_definitions(mm_engine, mm_meta, trained_model, encoders,
    #                                               classes=classes, dump_dir=dumps_dir)

    # candidates = json.load(open('{}/predicted_candidates.json'.format(dumps_dir), mode='rt'))
    candidates = json.load(open('{}/predicted_candidates-small.json'.format(dumps_dir), mode='rt'))

    # candidates.append(Candidate(timestamp_attribute_id=None,
    #                             activity_identifier_attribute_id=None,
    #                             relationship_id=None))

    shutil.copyfile(openslex_file_path, modified_mm)

    mm_engine_modif = ex.create_mm_engine(modified_mm)
    mm_meta_modif = ex.get_mm_meta(mm_engine_modif)

    ev.compute_events(mm_engine_modif, mm_meta_modif, candidates)

    breakpoint()


if __name__ == '__main__':
    test_candidates()
