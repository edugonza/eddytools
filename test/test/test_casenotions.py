import eddytools.extraction as ex
import eddytools.casenotions as cn
import json
import os

openslex_file_path = 'data/mm/metamodel-sample.slexmm'


def test_candidates():
    mm_engine = ex.create_mm_engine(openslex_file_path)
    rs = cn.get_relationships(mm_engine)
    assert rs.__len__() == 8
    for r in rs:
        assert r['rs'] in ['CONCERT_HALL_FK', 'SEAT_HALL_FK', 'BP_CONCERT_FK', 'BP_BAND_FK',
                           'TICKET_CONCERT', 'TICKET_SEAT', 'BOOKING_CON', 'BOOKING_FK']

    stats = cn.get_stats_mm(mm_engine)

    os.makedirs('output/dumps/', exist_ok=True)

    json.dump(stats, open('output/dumps/stats_mm.json', 'wt'), indent=True)

    candidates = cn.compute_candidates(mm_engine)

    json.dump(candidates, open('output/dumps/candidates.json', 'wt'), indent=True)

    assert candidates.__len__() > 0


if __name__ == '__main__':
    test_candidates()
