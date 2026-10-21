# Creating and Starting (Groups of) Experiments

You may have noticed the annotation of our main function in `main.py`:

```python
@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(config: DictConfig):

    # initialize, update, check configuration, and dump it to the newly created run directory
    establish_config(config)

    [...]
```

That's a [Hydra](https://hydra.cc/docs/intro/) decorator that allows us to run our main function with a supplied parameter configuration:
- The `config_path` parameter specifies the directory where the configuration files are located: `config/`, 
- The `config_name` specifies the name of the configuration file to use:  `config/base_config.yaml`.

[Hydra](https://hydra.cc/docs/intro/) is a powerful framework for managing configurations in Python applications, 
especially useful for machine learning experiments. 
It allows you to easily create, modify, and manage configurations without hardcoding them into your scripts.
Especially useful is the ability to create parameter groups and run multiple experiments with different configurations
or blocks of configurations.

Let's have a look at the start of our `config/base_config.yaml` file:

```yaml
defaults:
  - algorithm: default_algorithm
  - logging: default_logging
  - observations: default_obs
  - platform: local_cpu
  - scenario@train_scenario: default_scenario
  - scenario@validation_scenario: default_scenario
  - scenario@eval_scenario: default_scenario
  - shortest_paths: default_shortest_paths
  - visualization: default_vis  # turned off by default
  - _self_
  - override hydra/sweeper: list

[...]
```
- The `defaults` dictionary specifies pairs of configuration groups - group presets that will be loaded 
when the main function is executed. The configuration groups are covered by the respective subdirectories, 
whereas the specific values for each group are defined in the respective YAML files within that directory.
- The `group@new_name` syntax indicates that we load a preset from group `group`, 
and assign it a new name `new_name` in the resulting configuration.
- The `_self_` entry allows us to override the current configuration with additional parameters specified directly 
in the config file (below the `defaults` section).
- Finally, the `override` keyword allows us to modify specific parameters within that group. In this specific case, we are overriding the `hydra/sweeper` group to additionally allow a list of parameters for hyperparameter sweeps (https://github.com/ALRhub/hydra_list_sweeper).

NOTE: the resolve order is from top to bottom, so lower entries in the `defaults` section will override previous ones.

### Creating and starting a New Experiment

We separate configuration "preset" files and the actual configuration files for each experiment. 
The latter are located in the `config/_runs` directory, where you can freely add new YAML files for new experiments.
Be sure to include `# @package _global_` at the top of the file to ensure that the configuration parameters override
the global configuration defined in `base_config.yaml`. 
Within your new experiment configuration file, you can override any of the parameters
defined in the `base_config.yaml` file, or any of the configuration groups defined in the `config/` directory.
You can even combine multiple run configuration groups in a single experiment file, such as done in `config/_runs/dev25/sanity_check_algo.yaml`:

```yaml
# @package _global_

defaults:
  - _base_sanity_check
  - _self_

hydra:
  sweeper:
    grid_params:
      _runs/_algo: eigrp, magnneto, ospf, random_link_weight, random_next_hop, mslim, fieldlines
```

Here, we are using the `sanity_check` configuration group, which is defined in `config/_runs/_type/sanity_check.yaml`,
defining a very short "sanity check" scenario to test basic functionality of the framework. 
Then, we override the `hydra/sweeper` group to define a grid sweep over the `/_runs/_algo` parameter group,
which in turn uses run preset files defined in `config/_runs/_algo`.

Finally, to start the experiment, you can run the following command in your terminal:

```bash
python main.py +_runs/<your experiment file's' subdirectories stem within config/>=<your experiment file's name stem>
```

Here, the `+` sign indicates that we want to add a configuration group overriding some 
of the parameters defined in the `base_config.yaml` file. You can also override specific parameters 
without adding any prefix.

**Examples:**
```bash
python main.py +_runs=sanity_check_single
```

or

```bash
python main.py +_runs/dev25=sanity_check_algo logging.wandb.enabled=True
```

For further information on how to use Hydra, please refer to the [Hydra documentation](https://hydra.cc/docs/intro/).