config = {
    'fc1_dims' : 256,
    'fc2_dims' : 128,
    'batch_size' : 8,
    'num_workers': 8,
    'state_shape': (4,),
    'action_shape': (1,),
    'rollout_length': 32,
    'epochs' : 8,
    'std' : 0.01,
    'gamma': 0.8,
    'tau' : 0.8,
    'lr': 1e-5,
    'action_limit' : 1,
    'adv_norm' : False,
    'epsilon' : 0.1,
    'debug' : False
}
