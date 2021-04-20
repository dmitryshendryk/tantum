# Tantum

## Library with pipeline for Kaggle competitions
Supports GPU and TPU 

## Params

```
class CFG:
    OUTPUT_DIR = './'  # output dir of the artifacts
    TRAIN_PATH = '../input/' # input train path folder
    TEST_PATH = '../input/'  # input test path folder
    debug=True               # debug on small dataset
    apex=True                # auto precision mode  [True, Flase]
    device='GPU'             # device mode ['TPU', 'GPU']
    nprocs=1 # [1, 8]        # number of proc of TPU
    print_freq=100           # print frequency
    num_workers=4            # number of workers for DataLoader
    model_name='tf_efficientnet_b3_ns' # model name 
    size=512 # [224, 384, 512]         # image size
    freeze_epo = 0 # GradualWarmupSchedulerV2      # freeze epoch
    warmup_epo = 1 # GradualWarmupSchedulerV2      # warmup epochs
    cosine_epo = 9 # GradualWarmupSchedulerV2      # train epochs
    epochs = freeze_epo + warmup_epo + cosine_epo  #  total epochs
    scheduler='GradualWarmupSchedulerV2' # scheduler
    criterion='CrossEntropyLoss' # criterion
    T_0=10 # CosineAnnealingWarmRestarts  # step to renew CosineAnnealingWarmRestarts
    lr=1e-4                               # learning rate
    min_lr=1e-6                           # min learning rate 
    batch_size=16 #[32, 64]               # batch size
    weight_decay=1e-6                     # weight_decay
    gradient_accumulation_steps=1         # gradient accumulation
    max_grad_norm=1000
    rand_augment=True
    N=3 # RandAugment
    M=11 # RandAugment
    seed=2021
    target_size=5
    target_col='label'
    n_fold=5
    trn_fold=[0] #[0, 1, 2, 3, 4]
    train=True
    smoothing=0.05
    t1=0.3 # bi-tempered-loss 
    t2=1.0 # bi-tempered-loss 
    swa_start = 5                        # when start swa
    swa = True                           # stochastik weighted average
    cutmix=False                         # aug cutmix
    fmix=False                           # aug fmix
```


