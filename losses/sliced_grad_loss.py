import torch
import torch.nn as nn
import torch.nn.functional as F

class SlicedGradLoss:
    def __init__(self, ce_coef = 0.35, grad_coef=None, num_particles=1):
        self.ce_criterion = nn.CrossEntropyLoss()

        self.ce_coef = ce_coef
        self.grad_coef = grad_coef if grad_coef is not None else 1 - self.ce_coef

        self.num_particles = num_particles

    @torch.enable_grad()
    def __call__(self, X_clean, logits_clean, logits_denoised, Y):
        ce_loss = self.ce_criterion(logits_denoised, Y)

        vectors = torch.randn_like(
            X_clean.unsqueeze(0).expand(self.num_particles, *X_clean.shape)
        )
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True)

        grad_loss = torch.tensor(0)
        if self.grad_coef != 0:
            output_sum = logits_clean.sum()

            grad = torch.autograd.grad(
                outputs = output_sum,
                inputs = X_clean,
                create_graph = True
            )[0]
            grad = grad.unsqueeze(dim=0)

            # grad_loss = torch.norm(grad, p=2)
            grad_loss = torch.sum(grad * vectors) ** 2 / self.num_particles

        loss = self.ce_coef * ce_loss + self.grad_coef * grad_loss

        return loss, ce_loss, grad_loss
