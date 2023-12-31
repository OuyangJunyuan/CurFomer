from rd3d import PROJECT_ROOT
from easydict import EasyDict as dict

OPTIMIZATION = dict(
    type='AdamW',
    lr=0.01,
    weight_decay=0.01,
    betas=(0.9, 0.99)
)
LR = dict(
    type='OneCycleLR',
    max_lr=OPTIMIZATION.lr,
    total_steps='${total_steps}',
    pct_start=0.4,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=10,
    final_div_factor=1e4,
)
RUN = dict(
    max_epochs=20,
    samples_per_gpu=2,
    workers_per_gpu=4,
    grad_norm_clip=10.0,

    workflows=dict(
        train=[dict(state='train', split='train', epochs=20),
               dict(state='test', split='test', epochs=1)],
        test=[dict(state='test', split='test', epochs=1)]
    ),

    tracker=dict(
        interval={'train_info': 1, 'grad&param': 0.5},  # 'scenes': 0.5,
        metrics=[],
        # save_codes=dict(root=PROJECT_ROOT / 'rd3d', include=['py', 'cpp', 'cu', 'h']),
    ),
    checkpoints=dict(
        max=10,
        interval=1,
        model_zoo='model_zoo'
    ),
    
    custom_hooks=[]
)
