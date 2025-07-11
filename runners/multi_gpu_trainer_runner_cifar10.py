import os 
import numpy as np
import pandas as pd 
from tqdm import tqdm 

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data.distributed import DistributedSampler

from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights


from improved_diffusion.script_util import (
    create_gaussian_diffusion
)

from improved_diffusion.unet import (
    UNetModel
)

from improved_diffusion.resample import (
    create_named_schedule_sampler,
    ScheduleSampler
)

from losses import NoiseDenoiseAlignedLoss, GradLoss, SlicedGradLoss

from datasets import (
    ImageNet64Loader,
    TinyImageNetLoader
)

from utils import (
    compute_top_k,
    AverageMeter,
    resize_tensor,
    rest
)

from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers.models.swin.modeling_swin import SwinForImageClassification


class MultiGPUTrainerRunnerCifar10:
    def __init__(
        self, 
        data_config,
        diffusion_config,
        unet_config,
        classifier_config,
        train_config,
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        self.data_config = data_config
        self.diffusion_config = diffusion_config
        self.unet_config = unet_config
        self.classifier_config = classifier_config
        self.train_config = train_config

        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

        checkpoint = None
        self.ckpt_training_report = None 
        if self.train_config.checkpoint != None:
            checkpoint = self.load_checkpoint(self.train_config)
            self.ckpt_training_report = checkpoint["training_reports"]

        # self.device = torch.device(device)

        self.load_dataset()
        self.load_diffusion()
        self.load_unet()
        self.load_classifier(checkpoint = checkpoint)
        self.load_diffusion_scheduler()

        self.load_criterion()
        self.load_optimizer(checkpoint = checkpoint)
        self.load_lr_scheduler(checkpoint = checkpoint)
        self.load_grad_scaler()

        os.makedirs(self.train_config.log.path, exist_ok=True)
    
    def load_dataset(self):
        if self.data_config.data_name.lower() == "cifar10":
            from torchvision import transforms, datasets

            self.train_dataset = datasets.CIFAR10(
                self.data_config.train_path, 
                train=True, 
                download=True, 
                # transform = transforms.Compose([
                #     transforms.Resize(self.data_config.image_size),
                #     # transforms.RandomCrop(32, padding=4),
                #     # transforms.RandomHorizontalFlip(),
                #     transforms.ToTensor()
                # ])
                transform = self.data_config.input_transform
            )

            self.test_dataset = datasets.CIFAR10(
                self.data_config.test_path, 
                train=False, 
                download=True, 
                transform = transforms.Compose([
                    transforms.Resize(self.data_config.image_size),
                    transforms.ToTensor()
                ])
            )
        elif self.data_config.data_name == "imagenet_64":

            print("Loading training data...")
            self.train_dataset = ImageNet64Loader(
                data_root = self.data_config.train_path,
                transform = self.data_config.input_transform,
                dtype = self.data_config.dtype
            )

            print("Loading test data...")
            self.test_dataset = ImageNet64Loader(
                data_root = self.data_config.test_path,
                transform = self.data_config.input_transform,
                dtype = self.data_config.dtype
            )
        elif self.data_config.data_name == "tiny_imagenet":

            print("Loading training data...")
            self.train_dataset = TinyImageNetLoader(
                csv_path = self.data_config.train_path,
                transform = self.data_config.input_transform,
                dtype = self.data_config.dtype
            )

            print("Loading test data...")
            self.test_dataset = TinyImageNetLoader(
                csv_path = self.data_config.test_path,
                transform = self.data_config.input_transform,
                dtype = self.data_config.dtype
            )

        elif self.data_config.data_name.lower() == "imagenet":
            # from torchvision.datasets import ImageFolder
            
            # print("Loading training data...")
            # self.train_dataset = ImageFolder(
            #     root = self.data_config.train_path,
            #     transform = self.data_config.input_transform
            # )
            
            # print("Loading test data...")
            # self.test_dataset = ImageFolder(
            #     root = self.data_config.test_path,
            #     transform = self.data_config.input_transform
            # )

            from datasets import ImageNetLoader
            
            print("Loading training data...")
            self.train_dataset = ImageNetLoader(
                csv_path = self.data_config.train_path,
                transform = self.data_config.input_transform
            )
            
            print("Loading test data...")
            self.test_dataset = ImageNetLoader(
                csv_path = self.data_config.test_path,
                transform = self.data_config.input_transform
            )
        elif self.data_config.data_name.lower() == "svhn":
            from torchvision import transforms, datasets
            
            print("Loading training data...")
            self.train_dataset = datasets.SVHN(
                root = self.data_config.train_path,
                split = "train",
                download = True, 
                transform = self.data_config.input_transform
            )
            
            print("Loading test data...")
            self.test_dataset = datasets.SVHN(
                root = self.data_config.train_path,
                split = "test",
                download = True, 
                transform = self.data_config.input_transform
            )


        train_sampler = DistributedSampler(self.train_dataset)
        test_sampler = DistributedSampler(self.test_dataset)
        
        self.train_data_loader = DataLoader(
            self.train_dataset,
            batch_size = self.train_config.data_loader.batch_size,
            # shuffle = True,
            num_workers = self.train_config.data_loader.num_workers,
            drop_last = False,
            persistent_workers = True,
            pin_memory = True,
            sampler = train_sampler
        )

        self.test_data_loader = DataLoader(
            self.test_dataset,
            batch_size = self.train_config.data_loader.batch_size,
            # shuffle = False,
            num_workers = self.train_config.data_loader.num_workers,
            drop_last = False,
            persistent_workers = True,
            pin_memory = True,
            sampler = test_sampler
        )
    
    def load_diffusion(self):
        self.diffusion = create_gaussian_diffusion(
            steps = self.diffusion_config.diffusion_steps,
            learn_sigma = self.diffusion_config.learn_sigma,
            noise_schedule = self.diffusion_config.noise_schedule,
            use_kl = self.diffusion_config.use_kl,
            predict_xstart = self.diffusion_config.predict_xstart,
            rescale_timesteps = self.diffusion_config.rescale_timesteps,
            rescale_learned_sigmas = self.diffusion_config.rescale_learned_sigmas,
            timestep_respacing = self.diffusion_config.timestep_respacing
        )

    def load_unet(self):
        if self.train_config.cat_diff.diffusion_end_step == 0:
            self.unet = None 
            return 

        if self.data_config.data_name.lower() == "imagenet": 
            from guided_diffusion.unet import (
                UNetModel
            )
            self.unet = UNetModel(
                image_size = self.unet_config.unet_input_size,
                in_channels = self.unet_config.in_channels,
                model_channels = self.unet_config.model_channels,
                out_channels = self.unet_config.out_channels,
                num_res_blocks = self.unet_config.num_res_blocks,
                attention_resolutions = self.unet_config.attention_resolutions,
                dropout = self.unet_config.dropout,
                channel_mult = self.unet_config.channel_mult,
                num_classes = None,
                use_checkpoint = False,
                use_fp16 = self.unet_config.use_fp16,
                num_heads = self.unet_config.num_heads,
                num_head_channels = self.unet_config.num_head_channels,
                num_heads_upsample = self.unet_config.num_heads_upsample,
                use_scale_shift_norm = self.unet_config.use_scale_shift_norm,
                resblock_updown = self.unet_config.resblock_updown,
                use_new_attention_order = self.unet_config.use_new_attention_order,
            )

        else: 
            from improved_diffusion.unet import (
                UNetModel
            )
            self.unet = UNetModel(
                in_channels = self.unet_config.in_channels,
                model_channels = self.unet_config.model_channels,
                out_channels = self.unet_config.out_channels,
                num_res_blocks = self.unet_config.num_res_blocks,
                attention_resolutions = self.unet_config.attention_resolutions,
                dropout = self.unet_config.dropout,
                channel_mult = self.unet_config.channel_mult,
                num_classes = None,
                use_checkpoint = False,
                num_heads = self.unet_config.num_heads,
                num_heads_upsample = self.unet_config.num_heads_upsample,
                use_scale_shift_norm = self.unet_config.use_scale_shift_norm
            )

        self.unet.load_state_dict(
            torch.load(
                self.unet_config.pretrained_path,
                map_location = self.device
            )
        )

        if self.data_config.data_name.lower() == "imagenet":
            # self.unet = self.unet.to(self.device)
            self.convert_unet_to_fp16()
        else:
            self.unet = self.unet.to(self.device)
        # self.unet = self.unet.to(self.device, dtype=self.data_config.dtype)
        # self.unet.time_embed = self.unet.time_embed.to(dtype=self.data_config.dtype)
        # self.unet.time_embed = self.unet.time_embed.to(dtype=torch.float16)
        # self.unet.eval()

    def load_classifier(self, checkpoint = None):
        if self.classifier_config.name.lower() == "resnet50":
            self.classifier = resnet50(
                weights = ResNet50_Weights(ResNet50_Weights.DEFAULT)
            ).to(self.device)
        elif self.classifier_config.name.lower() == "resnet18":
            from torchvision.models import resnet18
            from torchvision.models import ResNet18_Weights
            self.classifier = resnet18(
                weights = ResNet18_Weights(ResNet18_Weights.DEFAULT)
            )
            if self.data_config.data_name == "cifar10":
                self.classifier.fc = torch.nn.Linear(512, 10)
            self.classifier = self.classifier.to(self.device)

        elif self.classifier_config.name.lower() == "resnet_v2":
            from archs.cifar_resnet import resnet
            
            self.classifier = resnet(
                depth = self.classifier_config.depth,
                num_classes = self.data_config.num_classes
            ).to(self.device)
        elif self.classifier_config.name.lower() == "swin_v2_b":
            from torchvision.models import swin_v2_b, Swin_V2_B_Weights
            import torch.nn as nn

            self.classifier = swin_v2_b(
                Swin_V2_B_Weights(
                    Swin_V2_B_Weights.IMAGENET1K_V1
                )
            )

            self.classifier.head = nn.Linear(
                in_features=1024, 
                out_features=self.data_config.num_classes, 
                bias=True
            )
            
            self.classifier = self.classifier.to(self.device)

        elif self.classifier_config.name.lower() == "vit-large":
            import timm

            self.classifier = timm.create_model('vit_large_patch32_224.orig_in21k', pretrained=True)
            self.classifier.reset_classifier(num_classes=self.data_config.num_classes)
            self.classifier = self.classifier.to(self.device)

            data_config = timm.data.resolve_model_data_config(self.classifier)
            input_transforms = timm.data.create_transform(**data_config, is_training=False)

            self.data_config.input_transform = input_transforms
            self.load_dataset()
        
        elif self.classifier_config.name.lower() == "vit-b":
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            processor = AutoImageProcessor.from_pretrained(
                "LucasThil/swin-tiny-patch4-window7-224-finetuned-tiny-imagenet"
            )
            self.classifier = AutoModelForImageClassification.from_pretrained(
                "LucasThil/swin-tiny-patch4-window7-224-finetuned-tiny-imagenet"
            ).to(self.device)

            self.data_config.input_transform = processor
            self.load_dataset()

        elif self.classifier_config.name.lower() == "beitv2":
            import timm

            self.classifier = timm.create_model(
                # 'beit_large_patch16_512',
                # "timm/beitv2_large_patch16_224.in1k_ft_in1k", 
                "timm/beitv2_base_patch16_224.in1k_ft_in1k",
                pretrained = True
            ).to(self.device)

            beit_data_config = timm.data.resolve_model_data_config(self.classifier)
            input_transforms = timm.data.create_transform(**beit_data_config, is_training=False)

            self.data_config.input_transform = input_transforms
            self.load_dataset()

        elif self.classifier_config.name.lower() == "beitv3":
            import timm

            self.classifier = timm.create_model(
                # 'beit_large_patch16_512',
                # "timm/beitv2_large_patch16_224.in1k_ft_in1k", 
                "beit_large_patch16_512",
                pretrained = True
            ).to(self.device)

            # beit_data_config = timm.data.resolve_model_data_config(self.classifier)
            # input_transforms = timm.data.create_transform(**beit_data_config, is_training=False)

            # self.data_config.input_transform = input_transforms
            # self.load_dataset()

        elif self.classifier_config.name.lower() == "vit_cifar10": 
            from transformers import AutoModelForImageClassification
            self.classifier = AutoModelForImageClassification.from_pretrained(
                "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
            ).to(self.device)

        else:
            print(f"Error!! Uknown classifier '{self.classifier_config.name}'")
            exit()
        
        if checkpoint != None:
            self.classifier.load_state_dict(
                checkpoint["classifier"]
            )

    def load_diffusion_scheduler(self):
        if self.train_config.cat_diff.diffusion_end_step == 0:
            self.schedule_sampler = None 
            return 
        elif self.train_config.cat_diff.sigma is not None:
            std_diffusion = self.diffusion.sqrt_one_minus_alphas_cumprod / self.diffusion.sqrt_alphas_cumprod
            T_end = np.argmin( 
                np.abs(self.train_config.cat_diff.sigma - std_diffusion)
            ) + 1
            T_start = T_end
            print(f"sigma: {self.train_config.cat_diff.sigma}\tT: {T_start}")
        
        else:
            T_end = None 
            T_start = 0
            if self.data_config.data_name.lower() == "imagenet":
                ts = list(self.diffusion.use_timesteps)
                ts.sort()
                ts = np.array(ts)
                nearest_t_index = np.argmin(
                    np.abs(
                        ts - self.train_config.cat_diff.diffusion_end_step
                    )
                )
                nearest_t = ts[nearest_t_index]

                print(f"DDIM Scheduler -> Nearest T -> Index: {nearest_t}, t: {nearest_t}")

                T_end = nearest_t_index

                self.train_config.cat_diff.diffusion_end_step = ts[nearest_t_index]

            else:
                T_end = self.train_config.cat_diff.diffusion_end_step

        class ModifiedUniformSampler(ScheduleSampler):
            def __init__(self, diffusion, end_step, start_step=0):
                self.diffusion = diffusion
                self._weights = np.ones([end_step])
                self._weights[:start_step-1] = 0

            def weights(self):
                return self._weights

        self.schedule_sampler = ModifiedUniformSampler(
            diffusion = self.diffusion,
            start_step = T_start,
            end_step = T_end
        )

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
        elif self.train_config.criterion.type.lower() == "sliced_grad_loss":
            from losses import SlicedGradLoss
            self.criterion = SlicedGradLoss(
                ce_coef = self.train_config.criterion.ce_coef, 
                grad_coef = self.train_config.criterion.grad_coef,
                num_particles = self.train_config.criterion.num_particles
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
        checkpoint_dict["unet_config"] = self.unet_config
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
        self.unet_config = checkpoint["unet_config"]
        self.classifier_config = checkpoint["classifier_config"]
        # self.train_config = checkpoint["train_config"]

        return checkpoint

    def load_grad_scaler(
        self,
    ):
        self.scaler = GradScaler()

    def convert_unet_to_fp16(self):
        self.unet = self.unet.to(self.device, dtype=torch.float16)
        self.unet.convert_to_fp16()

    def convert_unet_to_fp32(self): 
        self.unet = self.unet.to(self.device, dtype=torch.float32)
        self.unet.convert_to_fp32()
    
    def get_x_t(
        self,
        X, 
        t,
        noise = None
    ):
        X_t = None
        
        X_t = self.diffusion.q_sample(X, t, noise = noise)
        
        return X_t 
        # if noise is None:
        #     X_t = self.diffusion.q_sample(X, t, noise = self.universal_perturbation.repeat(X.shape[0], 1,1,1))
        # else:
        #     sqrt_alpha_t = self.diffusion.sqrt_alphas_cumprod[t.cpu()]
        #     sqrt_one_minus_alpha_t = self.diffusion.sqrt_one_minus_alphas_cumprod[t.cpu()]

        #     sqrt_alpha_t = torch.tensor(sqrt_alpha_t, device=self.device)[:, None, None, None]
        #     sqrt_one_minus_alpha_t = torch.tensor(sqrt_one_minus_alpha_t, device=self.device)[:, None, None, None]

        #     X_t = sqrt_alpha_t * X + sqrt_one_minus_alpha_t * noise
        
        # return X_t


    def get_x_0_t(
        self,
        X,
        t, 
        noise = None
    ):
        X = resize_tensor(X, self.unet_config.unet_input_size)

        X = 2 * X - 1

        X_t = self.get_x_t(
            X,
            t, 
            noise = noise
        ).to(dtype=self.data_config.dtype)

        # with torch.no_grad():
        X_0_t = self.diffusion.p_mean_variance(
            model = self.unet,
            x = X_t,
            t = t,
            clip_denoised = True,
        )["pred_xstart"]
        
        X = (X_0_t + 1) / 2
        
        return X

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
        X = resize_tensor(X, self.unet_config.unet_input_size)

        # Migrate X and Y to the selected device
        X = X.to(self.device) 

        Y = Y.long().to(self.device)

        # noise for X
        noise = torch.randn_like(X)
        noise_first = noise.clone()

        # criterions 
        mse_criterion = torch.nn.MSELoss()
        ce_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        for itr in range(self.train_config.adversary.instance_adversary_iterations):
            noise.requires_grad = True

            with autocast(enabled=False):
                self.convert_unet_to_fp32()
                X_t = self.diffusion.q_sample(X, t, noise = noise)

                X_0_t = self.diffusion.p_mean_variance(
                    model = self.unet,
                    x = X_t,
                    t = t,
                    clip_denoised = True,
                )["pred_xstart"]

                logits_denoised = self.classify(X_0_t, fp16_casting=False)

                loss = ce_criterion(logits_denoised, Y)

            self.scaler.scale(loss).backward()
            self.convert_unet_to_fp16()

            with torch.no_grad():
                torch.nn.utils.clip_grad_value_(noise, 0.5)
                grad = noise.grad
                # torch.nn.utils.clip_grad_value_(grad, 0.5)
                grad = torch.clip(grad, -1, 1)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print("Warning: nan/inf loss")
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"Warning: nan/inf gradient => nan: {torch.isnan(grad).any()}, inf: {torch.isinf(grad).any()}")

                # Gradient ascent
                noise = noise.detach() + self.train_config.adversary.optimizer.learning_rate * grad.sign()

                delta = torch.clamp(noise - noise_first, min=-self.train_config.adversary.epsilon, max=self.train_config.adversary.epsilon)
                
                noise = noise_first + delta
            
            noise.grad = None

        return noise

    def classify(
        self,
        X,
        fp16_casting = True
    ):
        with autocast(enabled = fp16_casting):
            X_processed = resize_tensor(X, self.classifier_config.classifier_input_size)
            # if self.data_config.data_name.lower() == "imagenet":
            #     X_processed = self.train_dataset.normalize(X_processed)

            if not fp16_casting:
                for name, param in self.classifier.named_parameters():
                    if param.dtype == torch.float16:
                        print(f"{name} => {param.dtype}")
                
            # logits_clean = self.classifier(X)
            logits = self.classifier(X_processed)

            return logits

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


        with autocast():
            X_denoised = X.clone()

            # Selecting t
            if self.train_config.cat_diff.active and train and self.schedule_sampler != None:
                noise = None

                t, _ = self.schedule_sampler.sample(X.shape[0], self.device)

                if adversary_perturbation:
                    noise = self.instance_adversary(X, Y, t)
                    self.classifier.train()
                
                X_denoised = self.get_x_0_t(X, t, noise=noise)
        
        # X = X.to(torch.float32)
        # X_denoised = X_denoised.to(torch.float32)
        
            # if self.classifier_config.name.lower() == "vit_cifar10":
            #     X = torch.nn.functional.interpolate(X, (224, 224), mode='bicubic', antialias=True)
            
            # Resize the classifier input 
            X = resize_tensor(X, self.classifier_config.classifier_input_size)
            X_denoised = resize_tensor(X_denoised, self.classifier_config.classifier_input_size)

            if isinstance(self.criterion, GradLoss) or isinstance(self.criterion, SlicedGradLoss):
                X.requires_grad_(True)
            
            # Forward
            logits_clean = self.classify(X)
            logits_denoised = self.classify(X_denoised)


            if isinstance(self.classifier, ViTForImageClassification) or isinstance(self.classifier, SwinForImageClassification):
                logits_clean = logits_clean.logits
                logits_denoised = logits_denoised.logits
            
            # Loss
            ce_loss = torch.tensor(0)
            kl_loss = torch.tensor(0)
            entropy_loss = torch.tensor(0)
            grad_loss = torch.tensor(0)

            if isinstance(self.criterion, NoiseDenoiseAlignedLoss):
                loss, ce_loss, kl_loss, entropy_loss = self.criterion(logits_clean, logits_denoised, Y)
            if isinstance(self.criterion, GradLoss) or isinstance(self.criterion, SlicedGradLoss):
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
            logits = logits_denoised if train and self.schedule_sampler != None else logits_clean,
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
        
        if self.local_rank == 0:
            t_train_data_loader = tqdm(total=len(self.train_data_loader))
            t_train_data_loader.set_description(f"Training @ Epoch={epoch}")
            
        # t_train_data_loader = tqdm(self.train_data_loader)
        # t_train_data_loader.set_description(f"Training @ Epoch={epoch}")

        self.optimizer.zero_grad()
        
        for batch_id, (batch) in enumerate(self.train_data_loader):
            # Forward the current batch
            loss, ce_loss, kl_loss, entropy_loss, grad_loss, batch_report = self.batch_forward(
                batch = batch,
                train = True,
                adversary_perturbation = self.train_config.adversary.adversary_perturbation
            )

            # loss.backward()
            self.scaler.scale(loss).backward()

            # Optimizing
            if ((batch_id+1) % self.train_config.optimizer.gradient_accumulation_steps == 0) or ((batch_id+1) == len(self.train_data_loader)):
                # self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
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

            if self.local_rank == 0:
                t_train_data_loader.update(1)
                t_train_data_loader.set_postfix({
                    "T": self.train_config.cat_diff.diffusion_end_step,
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

        if self.local_rank == 0:
            t_train_data_loader.close()

        return training_report, loss_meter.avg

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
        
        if self.local_rank == 0:
            t_eval_data_loader = tqdm(total=len(self.test_data_loader))
            if epoch!=None:
                t_eval_data_loader.set_description(f"Evaluation @ Epoch={epoch}")
            else:
                t_eval_data_loader.set_description(f"Evaluation")
        # t_eval_data_loader = tqdm(self.test_data_loader)
        # if epoch!=None:
        #     t_eval_data_loader.set_description(f"Evaluation @ Epoch={epoch}")
        # else:
        #     t_eval_data_loader.set_description(f"Evaluation")

        with torch.enable_grad():
            for batch_id, (batch) in enumerate(self.test_data_loader):
                
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

                if self.local_rank == 0:      
                    t_eval_data_loader.update(1)      
                    t_eval_data_loader.set_postfix({
                        "T": self.train_config.cat_diff.diffusion_end_step,
                        "lr": "{:.8f}".format(self.optimizer.param_groups[0]["lr"]),
                        "loss": loss_meter.avg,
                        "ce": ce_loss_meter.avg,
                        "kl": kl_loss_meter.avg,
                        "entropy": entropy_loss_meter.avg,
                        "grad": grad_loss_meter.avg,
                        "accuracy": accuracy_meter.avg, #accuracy_meter.avg,
                    })

        if self.local_rank == 0:
            t_eval_data_loader.close()
        return eval_report
    
    def train(self):

        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")

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
                file_name = f"{self.data_config.data_name}_{self.classifier_config.name.lower()}_{self.train_config.cat_diff.diffusion_end_step}_{epoch}.pt"
                self.save_checkpoint(
                    report,
                    f"{self.train_config.log.path}/{file_name}"
                )

            if (epoch + 1) % self.train_config.gpu_rest.freq == 0:

                rest(self.train_config.gpu_rest.sec)
            
            torch.distributed.barrier()

        file_name = f"{self.data_config.data_name}_{self.classifier_config.name.lower()}_{self.train_config.cat_diff.diffusion_end_step}_last.pt"
        self.save_checkpoint(
            report,
            f"{self.train_config.log.path}/{file_name}"
        )
                    


        