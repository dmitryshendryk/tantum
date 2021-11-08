import torch
import os

from torch.optim import optimizer

from tantum.model.arch import arch
from tantum.datasets import datasets
from tantum.utils.config import parse_commandline_args
from tantum.datasets.data_loaders import create_loaders_v2
from tantum.optimizer.optimizer import create_optimizer
from tantum.scheduler.scheduler import create_lr_scheduler
from tantum.trainer import mean_teacher

from tantum.model import get_model

build_model = {
    'mtv2': mean_teacher.Trainer
}

def run(config):
    print(config)
    print("pytorch version {}".format(torch.__version__))

    if config.save_freq!=0 and not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    print(config.dataset)

    dfconfig = datasets.load[config.dataset](config.num_labels)

    loaders = create_loaders_v2(**dfconfig, config=config)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = arch[config.arch](model_name=config.arch, target_size=dfconfig['num_classes'], pretrained=True)
    net = net.to(device)

    optimizer = create_optimizer(net.parameters(), config)
    scheduler = create_lr_scheduler(optimizer, config)

    ## run the model
    MTbased = set(['mt', 'ict'])
    if config.model[:-2] in MTbased or config.model[-5:]=='match':
        net2 = arch[config.arch](model_name=config.arch, target_size=dfconfig['num_classes'], pretrained=True)
        net2 = net2.to(device)
        trainer = build_model[config.model](net, net2, optimizer, device, config)
    else:
        trainer = build_model[config.model](net, optimizer, device, config)
    trainer.loop(config.epochs, *loaders, scheduler=scheduler)

if __name__ == '__main__':
    config = parse_commandline_args()
    run(config)