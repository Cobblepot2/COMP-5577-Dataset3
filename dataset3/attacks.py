"""Untargeted white-box attacks in the original [0,1] pixel space.

Callers supply an eval-mode model and validated settings. Gradients are taken
only with respect to input pixels; model parameter .grad is never populated.
"""
import torch
import torch.nn.functional as F


def fgsm(model, images, labels, epsilon):
    x = images.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), labels)
    gradient, = torch.autograd.grad(loss, x)
    return (x.detach() + epsilon * gradient.sign()).clamp(0, 1)


def pgd(model, images, labels, epsilon, step_size, steps, restarts=1,
        random_start=True, generator=None):
    """Keep per-image successful candidates across every iteration and restart.

    Prefer misclassification to correct classification, then higher CE within
    each group. This preserves found failures even if a later step recovers.
    Clean images are included, so already incorrect inputs remain failures.
    No FGSM candidates are injected into this PGD search.
    """
    original = images.detach()
    lower = (original - epsilon).clamp(0, 1)
    upper = (original + epsilon).clamp(0, 1)
    best = original.clone()
    with torch.no_grad():
        logits = model(best)
        best_loss = F.cross_entropy(logits, labels, reduction="none")
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
            losses = F.cross_entropy(logits, labels, reduction="none")
            with torch.no_grad():
                wrong = logits.argmax(1).ne(labels)
                better = (wrong & ~best_wrong) | ((wrong == best_wrong) & (losses > best_loss))
                best[better] = current[better]
                best_loss[better] = losses[better]
                best_wrong[better] = wrong[better]
            if iteration < steps:
                gradient, = torch.autograd.grad(losses.sum(), current)
                current = current.detach() + step_size * gradient.sign()
                current = torch.maximum(torch.minimum(current, upper), lower)
    return best.detach()
