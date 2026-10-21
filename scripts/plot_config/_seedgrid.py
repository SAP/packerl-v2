def is_module_init_seed_x(config, x):
    return (config["module_init_seed"] == x
            and config["train_algo_seed"] in train_algo_seeds
            and config["train_scen_seed"] in train_scen_seeds)

def is_train_algo_seed_x(config, x):
    return (config["train_algo_seed"] == x
            and config["module_init_seed"] in module_init_seeds
            and config["train_scen_seed"] in train_scen_seeds)

def is_train_scen_seed_x(config, x):
    return (config["train_scen_seed"] == x
            and config["module_init_seed"] in module_init_seeds
            and config["train_algo_seed"] in train_algo_seeds)

module_init_seeds = [0, 1, 2, 3]
train_algo_seeds = [0, 1, 2, 3, 4, 5]
train_scen_seeds = [0, 1, 2, 3, 4, 5]
module_init_bp_name = f"fixed module init seed \n(each col: {len(train_algo_seeds)}x{len(train_scen_seeds)} train sampling/train scenario seeds)"
train_algo_bp_name = f"fixed train sampling seed\n(each col: {len(module_init_seeds)}x{len(train_scen_seeds)} module init/train scenario seeds)"
train_scen_bp_name = f"fixed train scenario seed\n(each col: {len(module_init_seeds)}x{len(train_algo_seeds)} module init/train sampling seeds)"