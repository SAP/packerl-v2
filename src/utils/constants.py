"""
Non-configurable constants
"""

"""
Base directory for ns3
"""
NS3_DIR = "./ns3/"

"""
Base output directory for PackeRL
"""
DEFAULT_OUT_DIR = "../packerl_out"

"""
Name for the W&B project
"""
WANDB_PROJECT_NAME = "packerl"

"""
Pre-defined filenames for logging various metrics
"""
ROLLOUT_LOG_PER_EP_FILENAME = "rollout_per_episode.csv"
ROLLOUT_LOG_ALL_IT_FILENAME = "rollout_all_iterations.csv"
TRAINING_LOG_ALL_IT_FILENAME = "train_scalars_all_iterations.csv"
VALIDATION_LOG_PER_EP_FILENAME = "validation_per_episode.csv"
VALIDATION_LOG_ALL_IT_FILENAME = "validation_all_iterations.csv"
EVAL_LOG_ALL_EP_FILENAME = "evaluation_all_episodes.csv"
EVAL_LOG_PER_EP_FILENAME = "evaluation_per_episode.csv"

"""
Reference values for link weight calculations in OSPF and EIGRP
"""
OSPF_REF_DATARATE = 100000000  # OSPF reference datarate for calculating link weights (standard: 1e8)
EIGRP_REF_DATARATE = 10000000  # EIGRP reference datarate for calculating link weights (standard: 1e7)
EIGRP_REF_DELAY = 10  # EIGRP reference delay value for calculating link weights (standard: 10)
EIGRP_REF_MULTIPLIER = 256  # EIGRP reference multiplier for calculating link weights (standard: 256)

"""
Maximum Transmission Unit (MTU) of Ethernet in bytes
"""
ETHERNET_MTU = 1500

"""
Size of the IPv4 header in bytes
"""
IPV4_HEADER_SIZE = 20

"""
Size of the UDP header in bytes
"""
UDP_HEADER_SIZE = 8

"""
Size of the TCP header in bytes
"""
TCP_HEADER_SIZE = 40

"""
NS3-AI shared memory size in bytes
"""
SHM_SIZE = 2 ** 27  # 128 MiB

"""
ID of the "birdseye" observer (Global view over all agents without communication delay) 
"""
BIRDSEYE_OBSERVER_ID = -1
