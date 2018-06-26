import eddytools.extraction as ex
import eddytools.casenotions as cn
import json
import os
import shutil
from pprint import pprint

openslex_file_path_orig = 'data/mm/metamodel-sample.slexmm'
openslex_file_path = 'data/mm/metamodel-sample-modif.slexmm'


def test_candidates():

    shutil.copyfile(openslex_file_path_orig, openslex_file_path)

    mm_engine = ex.create_mm_engine(openslex_file_path)
    metadata = ex.get_mm_meta(mm_engine)

    rs = cn.get_relationships(mm_engine)
    assert rs.__len__() == 8
    for r in rs:
        assert r['rs'] in ['CONCERT_HALL_FK', 'SEAT_HALL_FK', 'BP_CONCERT_FK', 'BP_BAND_FK',
                           'TICKET_CONCERT', 'TICKET_SEAT', 'BOOKING_CON', 'BOOKING_FK']

    class_stats = cn.get_stats_mm(mm_engine)

    os.makedirs('output/dumps/', exist_ok=True)

    json.dump(class_stats, open('output/dumps/stats_mm.json', 'wt'), indent=True)

    candidates = cn.compute_candidates(mm_engine)

    json.dump(candidates, open('output/dumps/candidates.json', 'wt'), indent=True)

    candidates = json.load(open('output/dumps/candidates.json', 'rt'), object_hook=cn.CaseNotion)

    for idx, cand in enumerate(candidates):
        log_name = 'log_test_{}'.format(idx)
        print('Computing Log: {}'.format(log_name))
        log_id = cn.build_log_for_case_notion(mm_engine, cand,
                                              proc_name='proc_test_{}'.format(idx),
                                              log_name=log_name, metadata=metadata)

        sp = cn.compute_support_log(mm_engine, log_id, metadata)
        print('Support for {}: {}'.format(log_id, sp))
        ae = cn.compute_ae_log(mm_engine, log_id, metadata, sp)
        print('AE for {}: {}'.format(log_id, ae))
        lod = cn.compute_lod_log(mm_engine, log_id, metadata, sp)
        print('LoD for {}: {}'.format(log_id, lod))

        sp_lb = cn.compute_lb_support_cn(mm_engine, cand, metadata, class_stats)
        print('Lower Bound of Support for {}: {}'.format(idx, sp_lb))
        sp_ub = cn.compute_ub_support_cn(mm_engine, cand, metadata, class_stats)
        print('Upper Bound of Support for {}: {}'.format(idx, sp_ub))

        lod_lb = cn.compute_lb_lod_cn(mm_engine, cand, metadata, class_stats)
        print('Lower Bound of LoD for {}: {}'.format(idx, lod_lb))
        lod_ub = cn.compute_ub_lod_cn(mm_engine, cand, metadata, class_stats)
        print('Upper Bound of LoD for {}: {}'.format(idx, lod_ub))

        ae_lb = cn.compute_lb_ae_cn(mm_engine, cand, metadata, class_stats)
        print('Lower Bound of AE for {}: {}'.format(idx, ae_lb))
        ae_ub = cn.compute_ub_ae_cn(mm_engine, cand, metadata, class_stats)
        print('Upper Bound of AE for {}: {}'.format(idx, ae_ub))

        assert sp_ub >= sp >= sp_lb
        assert lod_ub >= lod >= lod_lb
        assert ae_ub >= ae >= ae_lb

    assert candidates.__len__() > 0


def test_stats():
    candidates = json.load(open('output/dumps/candidates.json', 'rt'), object_hook=cn.CaseNotion)
    class_stats = json.load(open('output/dumps/stats_mm.json', mode='rt'))

    mm_engine = None
    metadata = None

    bounds = cn.compute_bounds_of_candidates(candidates, mm_engine, metadata, class_stats)

    pred = cn.compute_prediction_from_bounds(bounds, 0.5, 0.5, 0.5)

    # for i, c in enumerate(candidates[:10]):
    #     sp_lb = bounds['sp_lb'][i]
    #     print('Lower Bound of Support for {}: {}'.format(i, sp_lb))
    #     sp_ub = bounds['sp_ub'][i]
    #     print('Upper Bound of Support for {}: {}'.format(i, sp_ub))
    #
    #     lod_lb = bounds['lod_lb'][i]
    #     print('Lower Bound of LoD for {}: {}'.format(i, lod_lb))
    #     lod_ub = bounds['lod_lb'][i]
    #     print('Upper Bound of LoD for {}: {}'.format(i, lod_ub))
    #
    #     ae_lb = bounds['ae_lb'][i]
    #     print('Lower Bound of AE for {}: {}'.format(i, ae_lb))
    #     ae_ub = bounds['ae_ub'][i]
    #     print('Upper Bound of AE for {}: {}'.format(i, ae_ub))

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

    pprint(params)

    ranking = cn.compute_ranking(pred, **params)
    pprint(ranking)

    print("Top-5")
    for i in ranking[:5]:
        pprint(candidates[i])


if __name__ == '__main__':
    test_candidates()
    #test_stats()
