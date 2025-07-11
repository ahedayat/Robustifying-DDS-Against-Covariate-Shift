import os 
import numpy as np
import pandas as pd 
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from diffusers import DDPMPipeline

import transformers

from losses import NoiseDenoiseAlignedLoss, GradLoss

from datasets import (
    ImageNet64Loader,
    TinyImageNetLoader
)

from utils import (
    compute_top_k,
    AverageMeter,
    resize_tensor
)

from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers.models.swin.modeling_swin import SwinForImageClassification


class MNISTTrainerRunner:
    def __init__(
        self, 
        data_config,
        diffusion_config,
        classifier_config,
        train_config,
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        self.data_config = data_config
        self.diffusion_config = diffusion_config
        self.classifier_config = classifier_config
        self.train_config = train_config

        checkpoint = None
        self.ckpt_training_report = None 
        if self.train_config.checkpoint != None:
            checkpoint = self.load_checkpoint(self.train_config)
            self.ckpt_training_report = checkpoint["training_reports"]

        self.device = torch.device(device)

        self.load_dataset()
        self.load_diffusion()
        self.load_classifier(checkpoint = checkpoint)
        self.load_t_noise()

        self.load_criterion()
        self.load_optimizer(checkpoint = checkpoint)
        self.load_lr_scheduler(checkpoint = checkpoint)

        os.makedirs(self.train_config.log.path, exist_ok=True)
    
    def load_dataset(self):
        self.train_dataset = MNIST(
            root = self.data_config.train_path, 
            train = True,
            download = True,
            transform = self.data_config.input_transform
        )

        self.test_dataset = MNIST(
            root = self.data_config.test_path, 
            train = False,
            download = True,
            transform = self.data_config.input_transform
        )

        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size = self.train_config.data_loader.batch_size,
            shuffle = True,
            num_workers = self.train_config.data_loader.num_workers,
            drop_last = False,
            persistent_workers = True,
            pin_memory = True
        )

        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size = self.train_config.data_loader.batch_size,
            shuffle = False,
            num_workers = self.train_config.data_loader.num_workers,
            drop_last = False,
            persistent_workers = True,
            pin_memory = True
        )
    
    def load_diffusion(self):
        self.diffusion_pipeline = DDPMPipeline.from_pretrained(
            self.diffusion_config.pretrained_path
        ).to(self.device)

    def load_classifier(self, checkpoint = None):
        if self.classifier_config.name.lower() == "beit":
            from transformers import AutoModelForImageClassification

            # processor = AutoImageProcessor.from_pretrained("Karelito00/beit-base-patch16-224-pt22k-ft22k-finetuned-mnist")
            self.classifier = AutoModelForImageClassification.from_pretrained(
                self.classifier_config.model_path
            ).to(self.device)

            # self.classifier = transformers.pipeline(
            #     "image-classification", 
            #     model = self.classifier_config.model_path
            # )

        elif self.classifier_config.name.lower() == "lenet": 
            from archs.lenet import LeNet
            self.classifier = LeNet().to(self.device)
            
        else:
            print(f"Error!! Uknown classifier '{self.classifier_config.name}'")
            exit()
        
        if checkpoint != None:
            self.classifier.load_state_dict(
                checkpoint["classifier"]
            )

    def load_t_noise(self):
        sigma = self.train_config.sigma 
        
        alpha_prod_t = self.diffusion_pipeline.scheduler.alphas_cumprod
        beta_prod_t = 1 - alpha_prod_t

        std_diffusion = (beta_prod_t / alpha_prod_t) ** 0.5

        self.diffusion_t = np.argmin( 
            torch.abs(sigma - std_diffusion)
        ) + 1

    def load_criterion(self):
        if self.train_config.criterion.type.lower() == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        
        elif self.train_config.criterion.type.lower() == "noise_denoise_aligned":
            self.criterion = NoiseDenoiseAlignedLoss(
                ce_coef = self.train_config.criterion.ce_coef,
                kl_coef = self.train_config.criterion.kl_coef, 
                entropy_coef = self.train_config.criterion.entropy_coef
            )
        elif self.train_config.criterion.type.lower() == "grad_loss":
            self.criterion = GradLoss(
                ce_coef = self.train_config.criterion.ce_coef, 
                grad_coef = self.train_config.criterion.grad_coef
            )

    def load_optimizer(self, checkpoint = None):
        self.optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr = self.train_config.optimizer.learning_rate,
            weight_decay = self.train_config.optimizer.weight_decay
        )

        if checkpoint != None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
    
    def load_lr_scheduler(self, checkpoint = None):
        if self.train_config.lr_scheduler.type == "reduce_on_plateau":
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                factor = 0.5, 
                patience = 2
            )
        elif self.train_config.lr_scheduler.type == "cosine_annealing":
            total_iterations = self.train_config.optimizer.num_epochs * len(self.train_data_loader)

            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max = total_iterations,
                eta_min = self.train_config.lr_scheduler.eta_min
            )

        if checkpoint != None:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def save_checkpoint(
        self,
        training_report,
        saving_path
    ):
        checkpoint_dict = dict()

        checkpoint_dict["classifier"] = self.classifier.state_dict()
        checkpoint_dict["optimizer"] = self.optimizer.state_dict()
        checkpoint_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        checkpoint_dict["training_reports"] = training_report
        checkpoint_dict["diffusion_config"] = self.diffusion_config
        checkpoint_dict["data_config"] = self.data_config
        checkpoint_dict["classifier_config"] = self.classifier_config
        checkpoint_dict["train_config"] = self.train_config

        torch.save(
            checkpoint_dict,
            saving_path
        )

    def load_checkpoint(
        self,
        train_config,
    ):
        checkpoint = torch.load(
            train_config.checkpoint,
            map_location = train_config.device
        )

        self.diffusion_config = checkpoint["diffusion_config"]
        # self.data_config = checkpoint["data_config"]
        # self.unet_config = checkpoint["unet_config"]
        self.classifier_config = checkpoint["classifier_config"]
        # self.train_config = checkpoint["train_config"]

        return checkpoint

    @torch.enable_grad()
    def get_x_t(
        self,
        X, 
        t,
        noise = None
    ):
        if noise is None: 
            noise = torch.randn_like(X)

        X_t = self.diffusion_pipeline.scheduler.add_noise(
            original_samples = X,
            noise = noise,
            timesteps = t,
        )
        
        return X_t 


    @torch.enable_grad()
    def get_x_0_t(
        self,
        X,
        t, 
        noise = None,
        clip = True
    ):
        X = resize_tensor(X, self.diffusion_config.unet_input_size)

        X = 2 * X - 1

        X_t = self.get_x_t(
            X,
            t, 
            noise = noise
        ).to(dtype=self.data_config.dtype)
        
        noise_pred = self.diffusion_pipeline.unet(X_t, t).sample
        
        alpha_prod_t = self.diffusion_pipeline.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        X_0_t = (X_t - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        if clip: 
            X_0_t = torch.clip(X_0_t, -1, 1)
        
        X = (X_0_t + 1) / 2
        
        return X

    def predict(
        self,
        X
    ):
        X_processed = X.clone()

        if self.classifier_config.name.lower() == "beit":
            X_processed = X_processed.repeat((1, 3, 1, 1))
        
        X_processed = resize_tensor(
            X_processed,
            self.classifier_config.classifier_input_size
        )
        
        X_processed = self.classifier_config.classifier_normalizer(X_processed)

        logits = self.classifier(X_processed)

        if isinstance(self.classifier, transformers.models.beit.modeling_beit.BeitForImageClassification):
            logits = logits.logits

        return logits

    def clip_projection(
        self,
        perturbation
    ):
        if self.train_config.adversary.epsilon != None:
            return torch.clip(
                perturbation,
                self.train_config.adversary.epsilon,
                - self.train_config.adversary.epsilon,
            )
        return perturbation

    def instance_adversary(
        self, 
        X,
        Y,
        t
    ):

        # Resize X
        X = resize_tensor(X, self.diffusion_config.unet_input_size)

        # Migrate X and Y to the selected device
        X = X.to(self.device) 

        Y = Y.long().to(self.device)

        # noise for X
        noise = torch.randn_like(X)
        noise_first = noise.clone()

        # criterions 
        mse_criterion = torch.nn.MSELoss()
        ce_criterion = torch.nn.CrossEntropyLoss()

        for itr in range(self.train_config.adversary.instance_adversary_iterations):
            noise.requires_grad = True

            X_0_t = self.get_x_0_t(
                X,
                t, 
                noise = noise,
                clip = True
            )

            # logits_clean = self.classifier(X)
            # logits_denoised = self.classifier(X_0_t)
            logits_denoised = self.predict(X_0_t)

            # p_clean = F.softmax(logits_clean, dim=1)
            # p_denoised = F.softmax(logits_denoised, dim=1)

            # ce_loss = ce_criterion(logits_denoised, Y)

            # kl_loss = F.kl_div(
            #     F.log_softmax(logits_denoised, dim=1), 
            #     p_clean, 
            #     reduction="batchmean"
            # )

            # Entropy: H(p) = -sum(p * log(p))
            # entropy_denoised = - (p_denoised * F.log_softmax(logits_denoised, dim=1)).sum(dim=1).mean()

            loss = ce_criterion(logits_denoised, Y)# - mse_criterion(X_0_t, X) 
            # loss = ce_loss + self.train_config.criterion.kl_coef * kl_loss + self.train_config.criterion.entropy_coef * entropy_loss
        
            grad = torch.autograd.grad(
                loss, 
                noise, 
                retain_graph=False, 
                create_graph=False
            )[0]

            # Gradient ascent
            noise = noise.detach() + self.train_config.adversary.optimizer.learning_rate * grad.sign()

            delta = torch.clamp(noise - noise_first, min=-self.train_config.adversary.epsilon, max=self.train_config.adversary.epsilon)
            
            noise = noise_first + delta

        return noise


    def batch_forward(
        self,
        batch,
        train = True,
        adversary_perturbation = False
    ):
        # Report
        report = pd.DataFrame()
        
        # Extracting X (augmented image) and Y (normal/abnormal) from batch
        X, Y = batch

        # Convert Y to long dtype
        Y = Y.long()

        # Migrate X and Y to the selected device
        X, Y = X.to(self.device), Y.to(self.device)

        X_denoised = X.clone()

        # Selecting t
        if self.train_config.covariate_shift_training and train:
            noise = None

            t = self.diffusion_t

            if adversary_perturbation:
                noise = self.instance_adversary(X, Y, t)
                self.classifier.train()
            
            X_denoised = self.get_x_0_t(X, t, noise=noise)
        
        # X = X.to(torch.float32)
        X_denoised = X_denoised.to(torch.float32)
        
        # if self.classifier_config.name.lower() == "vit_cifar10":
        #     X = torch.nn.functional.interpolate(X, (224, 224), mode='bicubic', antialias=True)
        
        # Resize the classifier input 
        # X = resize_tensor(X, self.classifier_config.classifier_input_size)
        # X_denoised = resize_tensor(X_denoised, self.classifier_config.classifier_input_size)

        if isinstance(self.criterion, GradLoss):
            X.requires_grad_(True)
            # X.requires_grad = True
        
        # Forward
        # logits_clean = self.classifier(X)
        # logits_denoised = self.classifier(X_denoised)
        # logits_clean = self.predict(X)

        logits_clean = self.predict(X)
        logits_denoised = self.predict(X_denoised)
        
        # Loss
        ce_loss = torch.tensor(0)
        kl_loss = torch.tensor(0)
        entropy_loss = torch.tensor(0)
        grad_loss = torch.tensor(0)

        # if isinstance(self.criterion, NoiseDenoiseAlignedLoss):
        #     loss, ce_loss, kl_loss, entropy_loss = self.criterion(logits_clean, logits_denoised, Y)
        if isinstance(self.criterion, GradLoss):
            loss, ce_loss, grad_loss = self.criterion(
                X_clean = X,
                logits_clean = logits_clean, 
                logits_denoised = logits_denoised, 
                Y = Y
            )
        else: 
            loss = self.criterion(logits_denoised, Y)


        # Accuracy
        accuracy = compute_top_k(
            logits = logits_denoised,
            labels = Y,
            k = 1,
            reduction = "mean"
        )

        # Report dataframe
        report = pd.DataFrame({
            "loss": [loss.detach().item()],
            "ce_loss": [ce_loss.detach().item() if ce_loss is not None else ce_loss],
            "kl_loss": [kl_loss.detach().item() if kl_loss is not None else kl_loss],
            "entropy_loss": [entropy_loss.detach().item() if entropy_loss is not None else entropy_loss],
            "grad_loss": [grad_loss.detach().item() if grad_loss is not None else grad_loss],
            "accuracy": [accuracy],
            "batch_size": [X.shape[0]]
        })

        return loss, ce_loss, kl_loss, entropy_loss, grad_loss, report

    def train_per_epoch(
        self,
        epoch
    ):
        training_report = pd.DataFrame()
        
        loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        entropy_loss_meter = AverageMeter()
        grad_loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        loss_meter.reset()
        ce_loss_meter.reset()
        kl_loss_meter.reset()
        entropy_loss_meter.reset()
        grad_loss_meter.reset()
        accuracy_meter.reset()
        
        t_train_data_loader = tqdm(self.train_data_loader)
        t_train_data_loader.set_description(f"Training @ Epoch={epoch}")

        self.optimizer.zero_grad()
        
        for batch_id, (batch) in enumerate(t_train_data_loader):
            # Forward the current batch
            loss, ce_loss, kl_loss, entropy_loss, grad_loss, batch_report = self.batch_forward(
                batch = batch,
                train = True,
                adversary_perturbation = self.train_config.adversary.adversary_perturbation
            )

            loss.backward()

            # Optimizing
            if ((batch_id+1) % self.train_config.optimizer.gradient_accumulation_steps == 0) or ((batch_id+1) == len(self.train_data_loader)):
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Updating loss meter
            loss_meter.update(loss.detach().item())
            ce_loss_meter.update(ce_loss.detach().item())
            kl_loss_meter.update(kl_loss.detach().item())
            entropy_loss_meter.update(entropy_loss.detach().item())
            grad_loss_meter.update(grad_loss.detach().item())

            # Updating accuracy meter
            accuracy_meter.update(batch_report.iloc[0]["accuracy"])

            # Adding batch_id to the batch report
            batch_report["batch_id"] = batch_id 

            # Adding epoch to the batch report
            batch_report["epoch"] = epoch

            # Adding batch report to the training report
            training_report = pd.concat([training_report, batch_report])


            t_train_data_loader.set_postfix({
                "T": self.diffusion_t,
                "lr": "{:.8f}".format(self.optimizer.param_groups[0]["lr"]),
                "loss": loss_meter.avg,
                "ce": ce_loss_meter.avg,
                "kl": kl_loss_meter.avg,
                "entropy": entropy_loss_meter.avg,
                "grad": grad_loss_meter.avg,
                "accuracy": accuracy_meter.avg,
            })

            # updating the learning rate scheduler if its type is cosine annealing
            if self.train_config.lr_scheduler.type == "cosine_annealing":
                self.lr_scheduler.step()

        return training_report, loss_meter.avg

    @torch.no_grad()
    def eval_per_epoch(
        self,
        epoch = None
    ):
        eval_report = pd.DataFrame()
        
        loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        entropy_loss_meter = AverageMeter()
        grad_loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()
        
        loss_meter.reset()
        ce_loss_meter.reset()
        kl_loss_meter.reset()
        entropy_loss_meter.reset()
        grad_loss_meter.reset()
        accuracy_meter.reset()
        
        t_eval_data_loader = tqdm(self.test_data_loader)
        if epoch!=None:
            t_eval_data_loader.set_description(f"Evaluation @ Epoch={epoch}")
        else:
            t_eval_data_loader.set_description(f"Evaluation")

        with torch.enable_grad():
            for batch_id, (batch) in enumerate(t_eval_data_loader):
                
                # Forward the current batch
                loss, ce_loss, kl_loss, entropy_loss, grad_loss, batch_report = self.batch_forward(
                    batch = batch,
                    train = False,
                    adversary_perturbation = False
                )
            
                # Updating loss meter
                loss_meter.update(loss.item())
                ce_loss_meter.update(ce_loss.detach().item())
                kl_loss_meter.update(kl_loss.detach().item())
                entropy_loss_meter.update(entropy_loss.detach().item())
                grad_loss_meter.update(grad_loss.detach().item())
            
                # Updating accuracy meter
                accuracy_meter.update(batch_report.iloc[0]["accuracy"])
            
                # Adding batch_id to the batch report
                batch_report["batch_id"] = batch_id 
            
                # Adding epoch to the batch report
                batch_report["epoch"] = epoch
            
                # Adding batch report to the training report
                eval_report = pd.concat([eval_report, batch_report])
            
                t_eval_data_loader.set_postfix({
                    "T": self.diffusion_t,
                    "lr": "{:.8f}".format(self.optimizer.param_groups[0]["lr"]),
                    "loss": loss_meter.avg,
                    "ce": ce_loss_meter.avg,
                    "kl": kl_loss_meter.avg,
                    "entropy": entropy_loss_meter.avg,
                    "grad": grad_loss_meter.avg,
                    "accuracy": accuracy_meter.avg, #accuracy_meter.avg,
                })
        return eval_report
    
    def train(self):

        report = pd.DataFrame()
        if self.train_config.checkpoint != None:
            report = self.ckpt_training_report

        for epoch in range(self.train_config.optimizer.num_epochs):

            # Training for one epoch
            training_report_epoch, average_training_loss = self.train_per_epoch(epoch)
            training_report_epoch["mode"] = "train"
            
            # Evaluating for one epoch
            eval_report_epoch = self.eval_per_epoch(epoch = epoch)
            eval_report_epoch["mode"] = "test"

            # updating report data frame
            report = pd.concat([report, training_report_epoch, eval_report_epoch])

            # updating learning rate if its type is 'reduce on plateau'
            if self.train_config.lr_scheduler.type == "reduce_on_plateau":
                self.lr_scheduler.step(average_training_loss)

            if (epoch+1) % self.train_config.log.freq == 0:
                file_name = f"MNIST_{self.classifier_config.name.lower()}_{self.diffusion_t.item()}_{epoch}.pt"
                self.save_checkpoint(
                    report,
                    f"{self.train_config.log.path}/{file_name}"
                )

        file_name = f"{self.data_config.data_name}_{self.classifier_config.name.lower()}_{self.diffusion_t.item()}_last.pt"
        self.save_checkpoint(
            report,
            f"{self.train_config.log.path}/{file_name}"
        )
                    


        