import eddytools.extraction as ex
import eddytools.events as ev
import os
import shutil
from pprint import pprint

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


def test_default_model(openslex=train_openslex_file_path, ground_truth=ground_truth_path):
    mm_engine = ex.create_mm_engine(openslex)
    mm_meta = ex.get_mm_meta(mm_engine)
    pred, candidates, aid = ev.discover_event_definitions(mm_engine, mm_meta, model='default')
    y_true = aid.load_y_true(candidates, ground_truth)
    scores = aid.score(y_true, pred)
    pprint(scores)


if __name__ == '__main__':
    print("Test Training")
    test_candidates()

    print("Test Prediction with Default")
    test_default_model()

    if os.path.isfile(mimic_file_path):
        print("Test Prediction with Default on MIMICIII")
        test_default_model(openslex=mimic_file_path, ground_truth=mimic_ground_truth)
