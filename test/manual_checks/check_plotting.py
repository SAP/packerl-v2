import sys
sys.path.append("../../scripts")

import numpy as np
import matplotlib.pyplot as plt

from src.utils.plotting import generate_plots


if __name__ == '__main__':
    np.random.seed(42)
    num_seeds = 16

    dummy_results = {
        "Setup 1": {
            "Metric A": {
                "Base": 380.54,
                "Other": 374.43,
                "Ours 1": np.random.normal(loc=380.54 * 1.03, scale=10, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=380.54 * 0.98, scale=2, size=num_seeds).tolist()
            },
            "Metric B": {
                "Base": 8.56,
                "Other": 8.76,
                "Ours 1": np.random.normal(loc=8.56 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=8.56 * 0.90, scale=0.1, size=num_seeds).tolist()
            },
            "Metric C": {
                "Base": 5.44,
                "Other": 5.40,
                "Ours 1": np.random.normal(loc=5.44 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=5.44 * 1.02, scale=0.1, size=num_seeds).tolist()
            },
            "Mean oscillation_ratio_global": {
                "Base": 0.0,
                "Other": 0.0,
                "Ours 1": np.random.uniform(0.0, 0.1, size=num_seeds).tolist(),
                "Ours 2": np.random.uniform(0.0, 0.1, size=num_seeds).tolist()
            }
        },
        "Setup 2": {
            "Metric A": {
                "Base": 450.5,
                "Other": 440.3,
                "Ours 1": np.random.normal(loc=450.0 * 1.03, scale=10, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=450.0 * 0.98, scale=2, size=num_seeds).tolist()
            },
            "Metric B": {
                "Base": 12.02,
                "Other": 12.2,
                "Ours 1": np.random.normal(loc=12.0 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=12.0 * 0.90, scale=0.1, size=num_seeds).tolist()
            },
            "Metric C": {
                "Base": 7.06,
                "Other": 6.8,
                "Ours 1": np.random.normal(loc=7.06 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=7.0 * 1.02, scale=0.1, size=num_seeds).tolist()
            },
            "Mean oscillation_ratio_global": {
                "Base": 0.0,
                "Other": 0.0,
                "Ours 1": np.random.uniform(0.0, 0.1, size=num_seeds).tolist(),
                "Ours 2": np.random.uniform(0.0, 0.1, size=num_seeds).tolist()
            }
        },
        "Setup 3": {
            "Metric A": {
                "Base": 300.25,
                "Other": 295.335,
                "Ours 1": np.random.normal(loc=300.23 * 1.03, scale=10, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=300.343 * 0.98, scale=2, size=num_seeds).tolist()
            },
            "Metric B": {
                "Base": 6.50,
                "Other": 6.79,
                "Ours 1": np.random.normal(loc=6.55 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=6.5 * 0.90, scale=0.1, size=num_seeds).tolist()
            },
            "Metric C": {
                "Base": 4.09,
                "Other": 3.90,
                "Ours 1": np.random.normal(loc=4.06 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=4.0 * 1.02, scale=0.1, size=num_seeds).tolist()
            },
            "Mean oscillation_ratio_global": {
                "Base": 0.0,
                "Other": 0.0,
                "Ours 1": np.random.uniform(0.0, 0.1, size=num_seeds).tolist(),
                "Ours 2": np.random.uniform(0.0, 0.1, size=num_seeds).tolist()
            }
        },
        "Setup 4": {
            "Metric A": {
                "Base": 500.26,
                "Other": 490.67,
                "Ours 1": np.random.normal(loc=500.25 * 1.03, scale=10, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=500.02 * 0.98, scale=2, size=num_seeds).tolist()
            },
            "Metric B": {
                "Base": 9.09,
                "Other": 9.11,
                "Ours 1": np.random.normal(loc=9.09 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=9.11 * 0.90, scale=0.1, size=num_seeds).tolist()
            },
            "Metric C": {
                "Base": 6.07,
                "Other": 5.8,
                "Ours 1": np.random.normal(loc=6.0 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=6.07 * 1.02, scale=0.1, size=num_seeds).tolist()
            },
            "Mean oscillation_ratio_global": {
                "Base": 0.0,
                "Other": 0.0,
                "Ours 1": np.random.uniform(0.0, 0.1, size=num_seeds).tolist(),
                "Ours 2": np.random.uniform(0.0, 0.1, size=num_seeds).tolist()
            }
        }
    }

    dummy_results_1 = {
        "Setup 1": {
            "Metric A": {
                "Base": 380.54,
                "Other": 374.43,
                "Ours 1": np.random.normal(loc=380.54 * 1.03, scale=10, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=380.54 * 0.98, scale=2, size=num_seeds).tolist()
            },
            "Metric B": {
                "Base": 8.56,
                "Other": 8.76,
                "Ours 1": np.random.normal(loc=8.56 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=8.56 * 0.90, scale=0.1, size=num_seeds).tolist()
            },
            "Metric C": {
                "Base": 5.44,
                "Other": 5.40,
                "Ours 1": np.random.normal(loc=5.44 * 0.95, scale=0.3, size=num_seeds).tolist(),
                "Ours 2": np.random.normal(loc=5.44 * 1.02, scale=0.1, size=num_seeds).tolist()
            },
            "Mean oscillation_ratio_global": {
                "Base": 0.0,
                "Other": 0.0,
                "Ours 1": np.random.uniform(0.0, 0.1, size=num_seeds).tolist(),
                "Ours 2": np.random.uniform(0.0, 0.1, size=num_seeds).tolist()
            }
        },
    }

    dummy_res_config = {
        "setups": ["Setup 1", "Setup 2", "Setup 3", "Setup 4"],
        "metrics": ["Metric A", "Metric B", "Metric C"],
        "units": {"Metric A": "MB", "Metric B": "ms", "Metric C": "%", "Mean oscillation_ratio_global": "%"},
        "lower_is_better": {"Metric A": False, "Metric B": True, "Metric C": True},
        "methods": ["Base", "Other", "Ours 1", "Ours 2"],
        "baseline_name": "Base",
        "num_rows": 2,
        "num_cols": 2,
        "plot_type": sys.argv[1],
    }

    dummy_res_config_1 = {
        "setups": ["Setup 1"],
        "metrics": ["Metric A", "Metric B", "Metric C"],
        "units": {"Metric A": "MB", "Metric B": "ms", "Metric C": "%", "Mean oscillation_ratio_global": "%"},
        "lower_is_better": {"Metric A": False, "Metric B": True, "Metric C": True},
        "methods": ["Base", "Other", "Ours 1", "Ours 2"],
        "baseline_name": "Base",
        "num_rows": 1,
        "num_cols": 1,
        "plot_type": sys.argv[1],
    }

    # if third argument is specified, add 'Mean oscillation_ratio_global' to the metrics
    if len(sys.argv) > 2:
        dummy_res_config["metrics"].append("Mean oscillation_ratio_global")
        dummy_res_config_1["metrics"].append("Mean oscillation_ratio_global")

    fig = generate_plots(dummy_results, dummy_res_config)
    # fig = generate_plots(dummy_results_1, dummy_res_config_1)
    plt.show()
