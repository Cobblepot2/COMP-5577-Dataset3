"""Verify saved attack predictions on the same CUDA numerical predictor."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
import json
import torch
import numpy as np
from dataset3.common import setup,load_records,write_json
from dataset3.evaluate_defense import load_predictor


def main():
    os.chdir(REPO)
    config,device=setup('configs/experiment.json')
    precision=dict(float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE'),
        cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    result=dict(precision=precision,precision_recorded_after_primary_runs=True,
        scope='Same unchanged Python/Torch/CUDA defaults and batch64 as primary runs; does not assert CPU or other-backend parity',models={})
    for name in ['baseline','hybrid','ensemble']:
        folder=Path('runs/cw_evaluation_'+name)
        report=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert report['completed']
        checkpoint=None if name=='baseline' else ('checkpoints/public85_hybrid_full37.pt' if name=='hybrid' else 'checkpoints/public85_residual_source.pt')
        secondary='checkpoints/public85_hybrid_full37.pt' if name=='ensemble' else None
        model,_,pipeline=load_predictor(config,checkpoint,device,name!='baseline',secondary,.5)
        assert pipeline==report['inference_pipeline']
        checks={}
        for scope in ['standard','budget']:
            payload=torch.load(folder/f'cw_{scope}_adversarial.pt',map_location='cpu',weights_only=True)
            for mode in ['float','quantized']:
                images=payload['adversarial']
                if mode=='quantized':images=(images*255).round()/255
                filename=f'cw_{scope}_'+('quantized_' if mode=='quantized' else '')+'predictions.csv'
                records=load_records(folder/filename,True)
                expected=np.array([int(r['label']) for r in records],dtype=np.int64)
                assert payload['ids']==[r['id'] for r in records]
                repeats=[]
                with torch.inference_mode():
                    for _ in range(2):
                        predictions=[]
                        for offset in range(0,len(images),config['batch_size']):
                            predictions.append(model(images[offset:offset+config['batch_size']].to(device)).argmax(1).cpu().numpy())
                        repeats.append(np.concatenate(predictions))
                differences=[int((pred!=expected).sum()) for pred in repeats]
                checks[f'{scope}_{mode}']=dict(differing_records_by_repeat=differences,
                    repeats_equal=bool(np.array_equal(*repeats)))
                if differences!=[0,0]:
                    raise AssertionError((name,scope,mode,differences))
        result['models'][name]=checks
        del model
    result['all_saved_predictions_reproduced_on_same_gpu']=True
    write_json('runs/cw_gpu_replay.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
