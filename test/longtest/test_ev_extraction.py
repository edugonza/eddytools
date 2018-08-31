from eddytools import events as ev
import json
import shutil


def extract_events():
    path_ev_def = 'data/mimic/ground_truth.json'
    path_mm_src = 'output/mimic/mm-src.slexmm'
    path_mm = 'output/mimic/mm-modif.slexmm'

    shutil.copyfile(path_mm_src, path_mm)

    ev_def = json.load(open(path_ev_def, mode='rt'))

    ev.extract_events(mm=path_mm, ev_def=ev_def)


if __name__ == '__main__':
    extract_events()
