
torchrun --nproc_per_node=2 ./train_cifar10_adversary_sigma_25.py
python rest.py --time 600
torchrun --nproc_per_node=2 ./train_cifar10_adversary_sigma_100.py
python rest.py --time 600
torchrun --nproc_per_node=2 ./train_cifar10_adversary_sigma_50.py
