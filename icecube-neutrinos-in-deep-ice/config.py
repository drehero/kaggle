import time


class config6:
    # io
    out_dir = "runs"
    in_dir = "data"
    eval_interval = 2000
    log_interval = 10
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    resume_training = False
    
    # wandb logging
    wandb_log = True
    wandb_project = "icecube"
    wandb_run_name = "run" + str(time.time())
    
    # data
    batch_size = 32
    max_len = 512
    
    # model
    n_layer = 6
    n_head = 12
    hidden_size = 768
    dropout = 0.0
    bias = False # do we use bias inside LayerNorm and Linear layers?
    
    padding_value = 0.0 

    seed = 1

    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    
    # system
    device = "cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "float16" # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

class config24:
    # io
    out_dir = "runs"
    in_dir = "data"
    eval_interval = 200
    log_interval = 10
    always_save_checkpoint = True # if True, always save a checkpoint after each eval
    resume_training = True
    
    # wandb logging
    wandb_log = True
    wandb_project = "icecube"
    wandb_run_name = "run" + str(time.time())
    
    # data
    batch_size = 256 
    max_len = 128 
    
    # model
    n_layer = 24 
    n_head = 16 
    hidden_size = 1024 
    dropout = 0.0
    bias = False # do we use bias inside LayerNorm and Linear layers?
    
    padding_value = 0.0 

    seed = 1

    # adamw optimizer
    learning_rate = 6e-4 # max learning rate
    max_iters = 600000 # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 2000 # how many steps to warm up for
    lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
    min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    
    # system
    device = "cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "float16" # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
