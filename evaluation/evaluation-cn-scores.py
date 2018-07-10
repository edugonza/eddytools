import shutil
from eddytools import casenotions as edcn
from eddytools.casenotions import CaseNotion
from eddytools import extraction as edex
import pandas as pd
from tqdm import tqdm
import json
import os
import pickle


def predict_cn(f_json: str):
    df: pd.DataFrame = pd.read_json(f_json)

    df_Y_sp = df['log_sp']
    df_Y_sp = df['log_lod']
    df_Y_sp = df['log_ae']


def evaluate_cn(mm_filepath: str, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    mm_filepath_tmp = '{}/mm-modif.slexmm'.format(output_dir)
    df_filepath = '{}/df.json'.format(output_dir)
    df_pickle_file = '{}/df.pickle'.format(output_dir)
    class_stats_filepath = '{}/class_stats.json'.format(output_dir)
    cands_filepath = '{}/candidates.json'.format(output_dir)

    shutil.copyfile(mm_filepath, mm_filepath_tmp)

    mm_engine = edex.create_mm_engine(mm_filepath_tmp)
    mm_meta = edex.get_mm_meta(mm_engine)

    cands = edcn.compute_candidates(mm_engine=mm_engine, max_length_path=5, cache_dir=output_dir)
    stats = {
        'proc_name': [],
        'log_name': [],
        'log_id': [],
        'log_sp': [],
        'log_lod': [],
        'log_ae': [],
        'cn_sp_lb': [],
        'cn_sp_ub': [],
        'cn_lod_lb': [],
        'cn_lod_ub': [],
        'cn_ae_lb': [],
        'cn_ae_ub': [],
        'num_classes': [],
        'e': [],
        'ir': [],
    }

    class_stats = edcn.get_stats_mm(mm_engine, mm_meta)

    json.dump(class_stats, open(class_stats_filepath, mode='wt'), indent=True)

    cn: CaseNotion
    for i, cn in enumerate(tqdm(cands.values(), total=len(cands), desc='Candidates')):
        proc_name = 'proc-{}'.format(i)
        log_name = 'log-{}'.format(i)
        log_id = edcn.build_log_for_case_notion(
            mm_engine, cn, proc_name, log_name, mm_meta)
        log_sp = edcn.compute_support_log(mm_engine, log_id, mm_meta)
        log_lod = edcn.compute_lod_log(mm_engine, log_id, mm_meta)
        log_ae = edcn.compute_ae_log(mm_engine, log_id, mm_meta)
        cn_sp_lb = edcn.compute_lb_support_cn(mm_engine, cn, mm_meta, class_stats)
        cn_sp_ub = edcn.compute_ub_support_cn(mm_engine, cn, mm_meta, class_stats)
        cn_lod_lb = edcn.compute_lb_lod_cn(mm_engine, cn, mm_meta, class_stats)
        cn_lod_ub = edcn.compute_ub_lod_cn(mm_engine, cn, mm_meta, class_stats)
        cn_ae_lb = edcn.compute_lb_ae_cn(mm_engine, cn, mm_meta, class_stats)
        cn_ae_ub = edcn.compute_ub_ae_cn(mm_engine, cn, mm_meta, class_stats)
        stats['proc_name'].append(proc_name)
        stats['log_name'].append(log_name)
        stats['log_id'].append(log_id)
        stats['log_sp'].append(log_sp)
        stats['log_lod'].append(log_lod)
        stats['log_ae'].append(log_ae)
        stats['cn_sp_lb'].append(cn_sp_lb)
        stats['cn_sp_ub'].append(cn_sp_ub)
        stats['cn_lod_lb'].append(cn_lod_lb)
        stats['cn_lod_ub'].append(cn_lod_ub)
        stats['cn_ae_lb'].append(cn_ae_lb)
        stats['cn_ae_ub'].append(cn_ae_ub)
        stats['num_classes'].append(cn.get_classes_ids().__len__())
        num_e = 0
        sum_e_per_o = 0
        for c_id in cn.get_classes_ids():
            num_e = num_e + class_stats[str(c_id)]['e']
            e_per_o = class_stats[str(c_id)]['ev_o']
            num_o = class_stats[str(c_id)]['o_w_ev']
            sum_e_per_o = float(sum_e_per_o) + float(float(e_per_o) / float(max(num_o, 1)))
        stats['e'].append(num_e)
        stats['ir'].append(sum_e_per_o / cn.get_classes_ids().__len__())

    df = pd.DataFrame(stats)

    df.to_json(df_filepath)
    pickle.dump(df, open(df_pickle_file, mode='wb'))


if __name__ == '__main__':

    evaluate_cn('data/mm/metamodel-sample.slexmm', output_dir='../private-data/RL/')

    # evaluate_cn('../private-data/ITV-mm.slexmm', output_dir='../private-data/ITV/')

    # evaluate_cn('../private-data/SAP-mm.slexmm', output_dir='../private-data/SAP/')
