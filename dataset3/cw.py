"""Explicit untargeted C&W-L2 search over pixels, never model parameters.

Reference: Carlini and Wagner, https://arxiv.org/abs/1608.04644 .
epsilon=None uses the standard [0,1] box. A numeric epsilon adds a per-input
Linf box; this is an explicitly constrained L2 variant, not original C&W-Linf.
"""
import math
import torch


def cw_margin(scores, labels, confidence=0.):
    real=scores.gather(1,labels[:,None]).squeeze(1)
    other=scores.clone()
    other.scatter_(1,labels[:,None],-torch.inf)
    return (real-other.amax(1)+confidence).clamp_min(0)


def cw_l2(model, images, labels, *, epsilon=None, steps=500,
          binary_search_steps=7, learning_rate=.01, initial_const=.001,
          confidence=0., abort_early=True, progress=None):
    """Minimize squared L2 + c*margin using tanh and Adam on an input modifier.

    Callers provide an eval-mode predictor and validated [0,1] float images.
    Per-image c search and minimum-distance successful candidates span all
    iterates/constants. Clean mistakes are retained at distance0. Score vectors
    may be raw logits or normalized log probabilities of a complete ensemble;
    the latter are logits for the final combined probability distribution.
    """
    if (steps<1 or binary_search_steps<1 or
        not math.isfinite(learning_rate) or learning_rate<=0 or
        not math.isfinite(initial_const) or initial_const<=0 or
        not math.isfinite(confidence) or confidence<0 or
        (epsilon is not None and (not math.isfinite(epsilon) or not 0<=epsilon<=1))):
        raise ValueError('Invalid C&W search settings')
    original=images.detach()
    with torch.no_grad():
        clean_scores=model(original)
        found=clean_scores.argmax(1).ne(labels)&cw_margin(clean_scores,labels,confidence).eq(0)
    best=original.clone()
    distance=torch.where(found,torch.zeros_like(labels,dtype=images.dtype),
                         torch.full_like(labels,torch.inf,dtype=images.dtype))
    history=[]
    if epsilon!=0:
        lower=torch.zeros_like(original) if epsilon is None else (original-epsilon).clamp(0,1)
        upper=torch.ones_like(original) if epsilon is None else (original+epsilon).clamp(0,1)
        half=(upper-lower)/2
        center=(upper+lower)/2
        base=torch.atanh(((original-center)/half).clamp(-1+1e-6,1-1e-6))
        c=torch.full_like(distance,initial_const)
        low=torch.zeros_like(c)
        high=torch.full_like(c,torch.inf)
        check_every=max(steps//10,1)
        for outer in range(binary_search_steps):
            modifier=torch.zeros_like(original,requires_grad=True)
            optimizer=torch.optim.Adam([modifier],lr=learning_rate)
            stage_success=torch.zeros_like(found)
            previous_loss=math.inf
            for iteration in range(steps+1):
                attacked=center+half*torch.tanh(base+modifier)
                scores=model(attacked)
                squared=(attacked-original).flatten(1).square().sum(1)
                margin=cw_margin(scores,labels,confidence)
                with torch.no_grad():
                    success=scores.argmax(1).ne(labels)&margin.eq(0)
                    stage_success|=success
                    better=success&(squared<distance)
                    best[better]=attacked[better]
                    distance[better]=squared[better]
                    found|=success
                loss=(squared+c*margin).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite C&W objective')
                if iteration==steps:
                    break
                if abort_early and iteration%check_every==0:
                    current_loss=loss.detach().item()
                    if current_loss>previous_loss*.9999:
                        break
                    previous_loss=current_loss
                gradient,=torch.autograd.grad(loss,modifier)
                optimizer.zero_grad(set_to_none=True)
                modifier.grad=gradient
                optimizer.step()
            with torch.no_grad():
                high=torch.where(stage_success,torch.minimum(high,c),high)
                low=torch.where(stage_success,low,torch.maximum(low,c))
                c=torch.where(torch.isfinite(high),(low+high)/2,c*10)
            entry=dict(search=outer+1,iterations=iteration,
                       stage_successes=int(stage_success.sum().item()),
                       cumulative_successes=int(found.sum().item()),
                       next_const_min=c.min().item(),next_const_max=c.max().item())
            history.append(entry)
            if progress is not None:progress(entry)
    delta=(best-original).flatten(1)
    return best.detach(),dict(found=found.cpu().tolist(),
        l2=delta.norm(p=2,dim=1).cpu().tolist(),
        linf=delta.abs().amax(1).cpu().tolist(),search_history=history,
        objective='squared_l2 + c * max(true_score - max_other_score + confidence, 0)',
        candidate_selection='minimum squared L2 among successful float candidates; clean fallback for failures',
        box='[0,1]' if epsilon is None else 'intersection of [0,1] and original +/- epsilon')
