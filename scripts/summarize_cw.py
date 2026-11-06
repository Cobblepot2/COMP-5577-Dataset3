"""Archive completed frozen200 evaluations; recompute CSV and tensor evidence."""
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import os
import csv
import json
import shutil
import hashlib
import numpy as np
import torch
from dataset3.common import ImageRecords,load_records,write_json
from dataset3.evaluate_defense import fixed_mixed_mask,attack_metrics
from dataset3.metrics import classification_metrics
from scripts.evaluate_public85_checks import select_records


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    os.chdir(REPO)
    rows,sampling=select_records(load_records('splits/validation.csv',True),'stress')
    dataset=ImageRecords(rows,Path('data/dataset3/train_images'),True)
    images=dataset.images.float()/255;labels=dataset.labels;mask=fixed_mixed_mask(200)
    archive=Path('docs/results/cw')
    archive.mkdir(parents=True,exist_ok=True)
    summary=dict(records=200,per_class_counts=sampling['per_class_counts'],
        selected_ids_sha256=sampling['selected_ids_sha256'],mixed_attacked_records=int(mask.sum()),
        interpretation='Frozen-model in-sample diagnostics only; finite untargeted search, no robustness certification or new Kaggle score',
        standard_scope='C&W-L2 in [0,1]; may exceed project Linf budget',
        constrained_scope='Same C&W-L2 objective with extra per-pixel Linf8/255 box, not original C&W-Linf algorithm',
        rows=[],original_model_training_performed=False)
    for name in ['baseline','hybrid','ensemble']:
        folder=Path(f'runs/cw_evaluation_{name}')
        report=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert report['completed'] and report['records']==200
        assert report['sampling']['selected_ids']==dataset.ids
        assert report['model_state_sha256_before']==report['model_state_sha256_after']
        assert report['checkpoint_sha256_before']==report['checkpoint_sha256_after']
        assert report['parameter_gradients_unpopulated'] and not report['model_training_performed']
        for path,digest in report['checkpoint_sha256_before'].items():assert sha(path)==digest
        def predictions(filename):
            records=load_records(folder/filename,True)
            assert [r['id'] for r in records]==dataset.ids
            return np.array([r['label'] for r in records],dtype=np.int64)
        clean=predictions('clean_predictions.csv');pgd=predictions('pgd20x3_predictions.csv')
        assert abs(classification_metrics(labels,pgd)['macro_f1']-report['same200_pgd20x3']['float_metrics']['macro_f1'])<1e-12
        for scope,entry in zip(['standard','budget'],report['attacks']):
            payload=torch.load(folder/f'cw_{scope}_adversarial.pt',map_location='cpu',weights_only=True)
            assert payload['ids']==dataset.ids and np.array_equal(payload['labels'].numpy(),labels)
            adv=payload['adversarial'];delta=(adv-images).flatten(1)
            assert torch.isfinite(adv).all() and adv.min()>=0 and adv.max()<=1
            if scope=='budget':assert delta.abs().max()<=8/255+1e-6
            assert abs(delta.abs().max().item()-entry['observed_linf'])<1e-12
            pred=predictions(f'cw_{scope}_predictions.csv')
            quant=predictions(f'cw_{scope}_quantized_predictions.csv')
            check=attack_metrics(labels,clean,pred,quant,mask)
            for key in ['float_metrics','quantized_8bit_metrics','mixed_50pct_metrics','mixed_50pct_quantized_8bit_metrics']:
                assert check[key]==entry[key],(name,scope,key)
            assert check['attack_successes_on_initially_correct']==entry['attack_successes_on_initially_correct']
        standard,budget=report['attacks']
        cw=predictions('cw_budget_predictions.csv')
        row=dict(model=name,clean_f1=report['clean']['macro_f1'],initially_correct=report['initially_correct'],
            pgd20x3_f1=report['same200_pgd20x3']['float_metrics']['macro_f1'],
            standard_cw_f1=standard['float_metrics']['macro_f1'],standard_cw_asr=standard['attack_success_rate_on_initially_correct'],
            standard_cw_outside_project_budget=standard['outputs_exceeding_project_linf'],
            standard_cw_max_linf=standard['observed_linf'],standard_cw_success_median_l2=standard['successful_initially_correct_l2']['median'],
            bounded_cw_f1=budget['float_metrics']['macro_f1'],bounded_cw_asr=budget['attack_success_rate_on_initially_correct'],
            bounded_cw_mixed_f1=budget['mixed_50pct_metrics']['macro_f1'],bounded_cw_8bit_f1=budget['quantized_8bit_metrics']['macro_f1'],
            bounded_cw_max_linf=budget['observed_linf'],
            cw_new_failures_beyond_pgd20=int(((clean==labels)&(pgd==labels)&(cw!=labels)).sum()),
            pgd20_failures_not_found_by_cw=int(((clean==labels)&(pgd!=labels)&(cw==labels)).sum()),
            joint_clean_pgd20_cw_accuracy=float(((clean==labels)&(pgd==labels)&(cw==labels)).mean()),
            standard_seconds=standard['seconds'],bounded_seconds=budget['seconds'],
            standard_peak_allocated_mib=standard['peak_allocated_mib'],bounded_peak_allocated_mib=budget['peak_allocated_mib'],
            run_metrics_sha256=sha(folder/'metrics.json'),model_unchanged=True)
        summary['rows'].append(row)
        shutil.copyfile(folder/'metrics.json',archive/f'{name}.json')
        for path in folder.glob('*predictions.csv'):
            shutil.copyfile(path,archive/f'{name}_{path.name}')
    csv_path=REPO.parent.parent/'outputs/public85/submission_public85_residual_hybrid50_flip.csv'
    summary['original_submission_sha256']=sha(csv_path)
    assert summary['original_submission_sha256']=='180c2302fc163c247d5e7b2e7e36be46dfc80ccab446ecb385afd021f4b750c4'
    baseline=json.loads((archive/'baseline.json').read_text(encoding='utf-8'))
    if baseline.get('provenance_attached_after_run'):
        original_runner=Path('runs/cw_evaluation_baseline/evaluate_cw_run_source.py')
        assert sha(original_runner)==baseline['script_sha256']
        shutil.copyfile(original_runner,archive/'baseline_run_source.py')
    replay=json.loads(Path('runs/cw_gpu_replay.json').read_text(encoding='utf-8'))
    assert replay['all_saved_predictions_reproduced_on_same_gpu']
    shutil.copyfile('runs/cw_gpu_replay.json',archive/'gpu_replay.json')
    write_json(archive/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
