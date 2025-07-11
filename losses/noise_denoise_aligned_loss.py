import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseDenoiseAlignedLoss:
    def __init__(self, ce_coef = 0.35, kl_coef=0.35, entropy_coef=None):
        self.ce_criterion = nn.CrossEntropyLoss()

        self.ce_coef = ce_coef
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef if entropy_coef is not None else 1 - (self.ce_coef + self.kl_coef)

    def __call__(self, logits_clean, logits_denoised, Y):
        ce_loss = self.ce_criterion(logits_denoised, Y)

        p_clean = F.softmax(logits_clean, dim=1)
        p_denoised = F.softmax(logits_denoised, dim=1)

        # KL divergence
        kl_loss = F.kl_div(
            F.log_softmax(logits_clean, dim=1), 
            p_denoised, 
            reduction="batchmean"
        )
        # kl_loss = F.kl_div(
        #     F.log_softmax(logits_denoised, dim=1), 
        #     p_clean, 
        #     reduction="batchmean"
        # )

        # Entropy: H(p) = -sum(p * log(p))
        entropy = - (p_clean * F.log_softmax(logits_clean, dim=1)).sum(dim=1).mean()
        # entropy = - (p_denoised * F.log_softmax(logits_denoised, dim=1)).sum(dim=1).mean()

        # Combine the losses 
        loss = self.ce_coef * ce_loss + self.kl_coef * kl_loss + self.entropy_coef * entropy

        return loss, ce_loss, kl_loss, entropy
