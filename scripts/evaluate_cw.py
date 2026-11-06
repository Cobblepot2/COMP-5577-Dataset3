"""Frozen200 C&W diagnostics. Optimize inputs only; never train a classifier."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import argparse
import hashlib
import json
from time import perf_counter
import numpy as np
import torch
from dataset3.common import ImageRecords,load_records,setup,write_json,write_predictions,provenance
from dataset3.cw import cw_l2
from dataset3.evaluate_defense import load_predictor,attack_metrics,fixed_mixed_mask,evaluate_attack
from dataset3.evaluation import make_loader,predict_clean
from dataset3.metrics import classification_metrics
from scripts.evaluate_public85_checks import select_records,exposure_scope


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def distance_stats(values,mask):
    selected=values[mask]
    return dict(records=int(mask.sum()),minimum=float(selected.min()) if len(selected) else None,
                median=float(np.median(selected)) if len(selected) else None,
                maximum=float(selected.max()) if len(selected) else None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['baseline','hybrid','ensemble'],required=True)
    parser.add_argument('--scope',choices=['both','standard','budget'],default='both')
    parser.add_argument('--output')
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--binary-search-steps',type=int,default=7)
    args=parser.parse_args()
    os.chdir(REPO)
    config,device=setup('configs/experiment.json')
    split=Path(config['split_dir'])/'validation.csv'
    rows,sampling=select_records(load_records(split,True),'stress')
    dataset=ImageRecords(rows,Path(config['data_dir'])/'train_images',True)
    checkpoint=None if args.model=='baseline' else ('checkpoints/public85_hybrid_full37.pt' if args.model=='hybrid' else 'checkpoints/public85_residual_source.pt')
    second='checkpoints/public85_hybrid_full37.pt' if args.model=='ensemble' else None
    model,primary,pipeline=load_predictor(config,checkpoint,device,args.model!='baseline',second,.5)
    model.eval()
    digest_before=state_sha(model)
    files_before={member['checkpoint']:sha(member['checkpoint']) for member in pipeline['members']}
    out=Path(args.output or f'runs/cw_evaluation_{args.model}')
    out.mkdir(parents=True,exist_ok=True)
    clean,_,clean_timing=predict_clean(model,make_loader(dataset,config),device)
    mask=fixed_mixed_mask(len(dataset))
    write_predictions(out/'clean_predictions.csv',dataset.ids,clean)
    scope,full_refit=exposure_scope(pipeline)
    report=dict(model=args.model,scope=scope,contains_active_full_refit_member=full_refit,
        provenance=provenance(config),config_sha256=sha('configs/experiment.json'),
        sampling=sampling,validation_csv_sha256=sha(split),inference_pipeline=pipeline,
        clean=classification_metrics(dataset.labels,clean),clean_timing=clean_timing,
        initially_correct=int((clean==dataset.labels).sum()),records=200,mixed_attacked_records=int(mask.sum()),
        mixed_mask_seed=5578,mixed_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        model_state_sha256_before=digest_before,checkpoint_sha256_before=files_before,
        implementation_sha256={name:sha(REPO/'dataset3'/name) for name in ['cw.py','attacks.py','tta.py','evaluate_defense.py']},
        sampling_implementation_sha256=sha(REPO/'scripts/evaluate_public85_checks.py'),
        script_sha256=sha(__file__),attacks=[],completed=False,model_training_performed=False,
        kaggle_score_inferred=False,certified_robustness=False,
        reference='Carlini and Wagner https://arxiv.org/abs/1608.04644; explicit implementation, not a ready-made attack package')
    write_json(out/'metrics.json',report)
    pgd=dict(name='white_box_pgd20x3',threat_model='white_box',method='pgd',epsilon=8/255,step_size=2/255,steps=20,restarts=3)
    entry,pred,quantized=evaluate_attack(model,model,dataset,config,device,pgd,clean,mask)
    report['same200_pgd20x3']=entry
    write_predictions(out/'pgd20x3_predictions.csv',dataset.ids,pred)
    write_predictions(out/'pgd20x3_quantized_predictions.csv',dataset.ids,quantized)
    write_json(out/'metrics.json',report)
    images=dataset.images.float()/255
    labels=torch.as_tensor(dataset.labels,dtype=torch.long)
    names=['standard','budget'] if args.scope=='both' else [args.scope]
    for name in names:
        epsilon=None if name=='standard' else 8/255
        settings=dict(steps=args.steps,binary_search_steps=args.binary_search_steps,
            learning_rate=.01,initial_const=.001,confidence=0.,abort_early=True,epsilon=epsilon)
        pieces=[];predictions=[];quantized_predictions=[];traces=[]
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=perf_counter()
        for offset in range(0,len(dataset),config['batch_size']):
            batch=images[offset:offset+config['batch_size']].to(device)
            target=labels[offset:offset+len(batch)].to(device)
            def progress(info):
                print(f"{args.model} {name} batch{offset//config['batch_size']+1} search{info['search']}: iterations={info['iterations']} success={info['cumulative_successes']}/{len(batch)}",flush=True)
            adversarial,trace=cw_l2(model,batch,target,progress=progress,**settings)
            with torch.inference_mode():
                predictions.append(model(adversarial).argmax(1).cpu().numpy())
                quantized_predictions.append(model((adversarial*255).round()/255).argmax(1).cpu().numpy())
            pieces.append(adversarial.cpu());traces.append(trace)
        torch.cuda.synchronize()
        seconds=perf_counter()-started
        allocated=torch.cuda.max_memory_allocated()/2**20
        reserved=torch.cuda.max_memory_reserved()/2**20
        attacked=torch.cat(pieces)
        pred=np.concatenate(predictions);quantized=np.concatenate(quantized_predictions)
        delta=(attacked-images).flatten(1)
        delta_q=((attacked*255).round()/255-images).flatten(1)
        l2=delta.norm(p=2,dim=1).numpy();linf=delta.abs().amax(1).numpy()
        linf_q=delta_q.abs().amax(1).numpy()
        if not (torch.isfinite(attacked).all() and attacked.min()>=0 and attacked.max()<=1):
            raise ValueError('C&W generated invalid pixels')
        if epsilon is not None and (linf.max()>epsilon+1e-6 or linf_q.max()>epsilon+1e-6):
            raise ValueError('Constrained C&W exceeded project budget')
        success=(clean==dataset.labels)&(pred!=dataset.labels)
        valid=linf<=8/255+1e-6;valid_q=linf_q<=8/255+1e-6
        metrics=attack_metrics(dataset.labels,clean,pred,quantized,mask)
        payload=out/f'cw_{name}_adversarial.pt'
        torch.save(dict(adversarial=attacked,ids=dataset.ids,labels=labels,settings=settings),payload)
        write_predictions(out/f'cw_{name}_predictions.csv',dataset.ids,pred)
        write_predictions(out/f'cw_{name}_quantized_predictions.csv',dataset.ids,quantized)
        metrics.update(name='cw_l2' if name=='standard' else 'cw_l2_with_additional_linf_box',settings=settings,
            attack_target='complete deployed predictor, untargeted, true-label margin',
            score_representation='raw logits for original model; normalized log probabilities serve as logits of the final averaged distribution for TTA/ensemble',
            observed_linf=float(linf.max()),observed_quantized_linf=float(linf_q.max()),
            observed_pixel_range=[attacked.min().item(),attacked.max().item()],
            successful_initially_correct_l2=distance_stats(l2,success),successful_initially_correct_linf=distance_stats(linf,success),
            initially_correct_successes_within_project_linf=int((success&valid).sum()),
            outputs_exceeding_project_linf=int((~valid).sum()),
            within_project_budget_filtered_metrics=attack_metrics(dataset.labels,clean,np.where(valid,pred,clean),np.where(valid_q,quantized,clean),mask),
            budget_filter_interpretation='Out-of-budget standard-L2 outputs use the clean prediction for this secondary count; search failure or invalidity is not evidence of robustness',
            seconds=seconds,images_per_second=len(dataset)/seconds,peak_allocated_mib=allocated,peak_reserved_mib=reserved,
            timing_scope='Warm-up excluded; input transfers, C&W input optimization, float/8bit predictor inference included; setup, metrics and payload/CSV writing excluded',
            quantization_scope='Post-hoc rounding of the minimum-L2 successful float candidate only; no quantization-aware search',
            payload=dict(path=str(payload),sha256=sha(payload)),search_traces=traces)
        report['attacks'].append(metrics)
        write_json(out/'metrics.json',report)
        print(f"FINISHED {args.model} {name}: F1={metrics['float_metrics']['macro_f1']:.6f} ASR={metrics['attack_success_rate_on_initially_correct']:.6f} maxLinf={linf.max():.6f} {seconds:.1f}s",flush=True)
    report.update(model_state_sha256_after=state_sha(model),checkpoint_sha256_after={p:sha(p) for p in files_before},
                  parameter_gradients_unpopulated=all(p.grad is None for p in model.parameters()))
    assert report['model_state_sha256_after']==digest_before and report['checkpoint_sha256_after']==files_before
    assert report['parameter_gradients_unpopulated']
    report['completed']=True
    write_json(out/'metrics.json',report)


if __name__=='__main__':main()
