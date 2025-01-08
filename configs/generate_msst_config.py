import argparse
from omegaconf import OmegaConf

def generate(cfg):
    msst_cfg = {
        'audio': {'chunk_size': 441000, 'min_mean_abs': 0.0, 'num_channels': 2, 'sample_rate': cfg.model.sr},
        'augmentations': {'enable': False},
        'inference': {'batch_size': 1, 'num_overlap': 4},
        'model': {
            'feature_dim': cfg.model.feature_dim,
            'layer': cfg.model.layer,
            'sr': cfg.model.sr,
            'win': cfg.model.win
        },
        'training': {
            'batch_size': 1, 'coarse_loss_clip': True, 'grad_clip': 0, 'instruments': ['restored', 'addition'], 
            'lr': 1.0, 'num_epochs': 1000, 'num_steps': 1000, 'optimizer': 'prodigy', 'patience': 2, 'q': 0.95, 
            'reduce_factor': 0.95, 'target_instrument': 'restored', 'use_amp': True
        }
    }
    return msst_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--original_config", type=str, default="configs/apollo.yaml")
    parser.add_argument("-o", "--output_config", type=str, default="configs/apollo_msst.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.original_config)
    msst_cfg = generate(cfg)

    with open(args.output_config, 'w') as f:
        OmegaConf.save(msst_cfg, f)
    print(f"Generated MSST config file at {args.output_config}")