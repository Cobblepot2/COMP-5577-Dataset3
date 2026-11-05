"""Explicit untargeted logit-margin PGD in the original pixel space."""
import torch


def logit_margin(logits, labels):
    """Maximize max(other-class logit) minus the true-class logit per image."""
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    return other.amax(1) - logits.gather(1, labels[:, None]).squeeze(1)


def margin_pgd(model, images, labels, epsilon, step_size, steps, restarts=1,
               random_start=True, generator=None):
    """Retain failures, then the highest margin, across all starts and iterates.

    Clean images participate in selection. Gradients are requested only for
    input pixels; model parameter .grad is never populated by the search.
    """
    original = images.detach()
    lower = (original - epsilon).clamp(0, 1)
    upper = (original + epsilon).clamp(0, 1)
    best = original.clone()
    with torch.no_grad():
        logits = model(best)
        best_objective = logit_margin(logits, labels)
        best_wrong = logits.argmax(1).ne(labels)
    for _ in range(restarts):
        if random_start:
            noise = torch.empty_like(original).uniform_(-epsilon, epsilon, generator=generator)
            current = (original + noise).clamp(0, 1)
        else:
            current = original.clone()
        for iteration in range(steps + 1):
            current = current.detach().requires_grad_(True)
            logits = model(current)
            objective = logit_margin(logits, labels)
            with torch.no_grad():
                wrong = logits.argmax(1).ne(labels)
                better = ((wrong & ~best_wrong) |
                          ((wrong == best_wrong) & (objective > best_objective)))
                best[better] = current[better]
                best_objective[better] = objective[better]
                best_wrong[better] = wrong[better]
            if iteration < steps:
                gradient, = torch.autograd.grad(objective.sum(), current)
                current = current.detach() + step_size * gradient.sign()
                current = torch.maximum(torch.minimum(current, upper), lower)
    return best.detach()
