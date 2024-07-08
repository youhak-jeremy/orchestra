# Specloss
python main.py --config_dict="{'train_mode': 'specloss', 'da_method': 'specloss', 'num_clients': 10, 'div_aware_update': True, 'stateful_client': True, 'local_bsize': 128, 'local_lr': 0.03, 'num_global_clusters': 64, 'num_local_clusters': 8, 'fraction_fit': 1.0}" > log_specloss_cifar10_cn10.txt

# Simsiam
python main.py --config_dict="{'train_mode': 'simsiam', 'da_method': 'simsiam', 'num_clients': 10, 'div_aware_update': True, 'stateful_client': True, 'local_bsize': 128, 'local_lr': 0.03, 'num_global_clusters': 64, 'num_local_clusters': 8, 'fraction_fit': 1.0}" > log_simsiam_cifar10_cn10.txt

# SimCLR
python main.py --config_dict="{'train_mode': 'simclr', 'da_method': 'simclr', 'num_clients': 10, 'div_aware_update': True, 'stateful_client': True, 'local_bsize': 128, 'local_lr': 0.03, 'num_global_clusters': 64, 'num_local_clusters': 8, 'fraction_fit': 1.0}" > log_simclr_cifar10_cn10.txt

