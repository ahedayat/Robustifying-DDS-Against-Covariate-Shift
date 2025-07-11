import torch
import torch.nn as nn
import torch.nn.functional as F

class GradLoss:
    def __init__(self, ce_coef = 0.35, grad_coef=None):
        self.ce_criterion = nn.CrossEntropyLoss()

        self.ce_coef = ce_coef
        self.grad_coef = grad_coef if grad_coef is not None else 1 - self.ce_coef

    @torch.enable_grad()
    def __call__(self, X_clean, logits_clean, logits_denoised, Y):
        ce_loss = self.ce_criterion(logits_denoised, Y)

        grad_loss = torch.tensor(0)
        if self.grad_coef != 0:
            output_sum = logits_clean.sum()

            grad = torch.autograd.grad(
                outputs = output_sum,
                inputs = X_clean,
                create_graph = True
            )[0]

            grad_loss = grad.view(grad.size(0), -1).norm(2, dim=1).mean()
            grad_loss = torch.log(1 + grad_loss)

            # grad_loss = torch.norm(grad, p=2)
            # grad_loss = torch.log(1 + torch.norm(grad, p=2))

        loss = self.ce_coef * ce_loss + self.grad_coef * grad_loss

        return loss, ce_loss, grad_loss
