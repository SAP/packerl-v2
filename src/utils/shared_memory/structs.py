"""
These structs define basic components of the entire simulation framework
such as network graphs and their parts, observation structures and
action spaces. As we're using ns3-ai and its shared memory paradigm, we have to
define all data structures that are accessed both from the simulator
side (C++) and the learning/control side (python) as structs on both
sides, and even take care of the member ordering to guarantee identical struct
sizes.

Consult shared-structs.h in the ns3-packerl module for a more detailed documentation
of all the structs' members.
"""
from typing import List
from ctypes import Structure, c_bool, c_double, c_uint32, c_uint64, c_float, c_long, sizeof, Union as cUnion

import numpy as np

"""
We can send arbitrary amounts of data between the C++ and Python side, but we have to
define a maximum number of elements for each data structure. This is due to the fact that
we have to allocate memory for these structures in the shared memory space, and we have to
know the maximum size of the memory block to allocate. If there is more data to send than
the maximum size, we have to split the data into multiple blocks and send them one after the
other.
"""
MAX_ELEMS = 64000


"""
Used to define the mapping between Ctypes and Numpy dtypes for conversion.
"""
NUMPY_DTYPE_CONVERSIONS = {
    c_uint32: np.uint32,
    c_uint64: np.uint64,
    c_double: np.float64,
    c_float: np.float32,
    c_bool: bool,
    c_long: np.int64
}


def struct_to_numpy_dtype(struct):
    """
    Converts a ctypes.Structure to a numpy dtype.
    :param struct: the ctypes.Structure
    :return: the numpy dtype
    """
    fields = struct._fields_
    names = [field[0] for field in fields]
    types = [NUMPY_DTYPE_CONVERSIONS[field[1]] for field in fields]

    # align=True is necessary to ensure that the dtype is compatible with the C++ struct
    return np.dtype(list(zip(names, types)), align=True)


class BaseStructure(Structure):
    """
    Base class for all shared memory structures.
    It is implemented as a Ctypes.Structure with integrated default values.
    """

    def __init__(self, **kwargs):
        """
        Initialize the structure.
        :param kwargs: values different to defaults
        :type kwargs: dict
        """
        values = type(self)._defaults_.copy()
        values.update(kwargs)

        super().__init__(**values)


class TopologyEdge(BaseStructure):
    """
    Models a network edge, which is a point-to-point communication channel characterized by
    datarate, 2 nodes, as well as channel delay in ms.
    """
    _fields_ = [
        ('fst', c_uint32),
        ('snd', c_uint32),
        ('datarate', c_uint64),
        ('delay', c_uint32),
    ]
    _defaults_ = {}


class TrafficFlowArrival(BaseStructure):
    """
    Models a traffic flow arrival event, which is characterized by a source and destination node,
    an amount of data to send, a datarate, and a flag indicating whether the traffic is TCP or not.
    In the 'upcomingEventTypes' array, this event is represented by the value 0.
    """
    _fields_ = [
        ('t', c_double),
        ('src', c_uint32),
        ('dst', c_uint32),
        ('amount', c_uint64),
        ('datarate', c_uint64),
        ('isTCP', c_bool),
    ]
    _defaults_ = {}


class LinkFailure(BaseStructure):
    """
    Models a link failure event, which is characterized by a timestamp and the two nodes that are
    connected by the failed link. In the 'upcomingEventTypes' array, this event is represented by the value 1.
    """
    _fields_ = [
        ('t', c_double),
        ('fst', c_uint32),
        ('snd', c_uint32),
    ]
    _defaults_ = {}


class Event(cUnion):
    """
    A union type that can hold either a TrafficFlowArrival or a LinkFailure event.
    """
    _fields_ = [
        ('trafficFlowArrival', TrafficFlowArrival),
        ('linkFailure', LinkFailure),
    ]
    _defaults_ = {}


class CommunicationDelay(BaseStructure):
    """
    Specifies a communication delay value between two routing nodes (src, dst).
    Used to assemble monitoring shapshots according to what remote nodes should see and what not (yet).
    """
    _fields_ = [
        ('src', c_uint32),
        ('dst', c_uint32),

        ('delay', c_double),
    ]
    _defaults_ = {}

comm_delay_dtype = struct_to_numpy_dtype(CommunicationDelay)


class GlobalRewardWeights(BaseStructure):
    """
    Models the global reward weights, which are used to calculate the global reward.
    """
    _fields_ = [
        ('txWeight', c_double),
        ('reTxWeight', c_double),
        ('inFlightWeight', c_double),
        ('rxWeight', c_double),
        ('deliveredWeight', c_double),
        ('ttlDroppedWeight', c_double),
        ('tcpDiscardedWeight', c_double),
        ('givenUpWeight', c_double),

        ('maxDelayWeight', c_double),
        ('avgDelayWeight', c_double),
        ('avgJitterWeight', c_double),
        ('avgTcpSenderSrttWeight', c_double),

        ('dropRatioWeight', c_double),
        ('inFlightRatioWeight', c_double),
        ('maxLinkUtilizationWeight', c_double),
        ('avgLinkUtilizationWeight', c_double),
        ('maxTxQueueRelativeLoadWeight', c_double),
        ('avgTxQueueRelativeLoadWeight', c_double),

        ('transmittedWeight', c_double),
        ('macTxDroppedWeight', c_double),
        ('phyRxDroppedWeight', c_double),
    ]
    _defaults_ = {}


class ObservedGlobal(BaseStructure):
    """
    A snapshot of the global network state from an observer's point of view.
    """
    _fields_ = [
        ('observerId', c_long),

        ('tx', c_uint64),
        ('reTx', c_uint64),
        ('inFlight', c_uint64),
        ('rx', c_uint64),
        ('delivered', c_uint64),
        ('ttlDropped', c_uint64),
        ('tcpDiscarded', c_uint64),
        ('givenUp', c_uint64),

        ('maxDelay', c_double),
        ('delayCount', c_uint32),
        ('avgDelay', c_double),
        ('jitterCount', c_uint32),
        ('avgJitter', c_double),
        ('tcpSenderSrttCount', c_uint32),
        ('avgTcpSenderSrtt', c_double),

        ('udpRemainingFlows', c_uint32),
        ('udpRemainingFlowBytes', c_uint64),
        ('udpRemainingFlowDatarateSum', c_uint64),
        ('tcpRemainingFlows', c_uint32),
        ('tcpRemainingFlowBytes', c_uint64),
        ('tcpRemainingFlowCWndSum', c_uint64),

        ('txQueueCapacity', c_uint64),
        ('txQueueLoad', c_uint64),

        ('transmitted', c_uint64),
        ('transmittable', c_uint64),
        ('macTxDropped', c_uint64),
        ('phyRxDropped', c_uint64),

        ('dropRatio', c_double),
        ('inFlightRatio', c_double),
        ('maxLinkUtilization', c_double),
        ('avgLinkUtilization', c_double),
        ('maxTxQueueRelativeLoad', c_double),
        ('avgTxQueueRelativeLoad', c_double),
    ]
    _defaults_ = {}


observed_global_dtype = struct_to_numpy_dtype(ObservedGlobal)
global_meta_idx = [observed_global_dtype.names.index('observerId')]


def get_accepted_global_feat_idx(acceptable_features: List[str]) -> List[int]:
    """
    Returns the indices of the accepted features in the observed global struct.
    :param acceptable_features: the accepted features
    :return: the indices of the accepted features
    """
    return sorted([observed_global_dtype.names.index(f) for f in acceptable_features])


class ObservedNode(BaseStructure):
    """
    Models a snapshot of an observed node from a specific observer's perspective,
    which is taken with respect to a certain amount
    of passed simulation time. Such snapshots hold network performance statistics within the
    given time frame, in this case specific to a node in the network.
    """
    _fields_ = [
        ('observerId', c_long),
        ('nodeId', c_uint32),

        ('txAsSender', c_uint64),
        ('txAsReceiver', c_uint64),
        ('reTxAsSender', c_uint64),
        ('reTxAsReceiver', c_uint64),
        ('inFlightAsSender', c_uint64),
        ('inFlightAsReceiver', c_uint64),
        ('rxAsSender', c_uint64),
        ('rxAsReceiver', c_uint64),
        ('deliveredAsSender', c_uint64),
        ('deliveredAsReceiver', c_uint64),
        ('ttlDroppedAsSender', c_uint64),
        ('ttlDroppedAsReceiver', c_uint64),
        ('ttlDroppedAsDropper', c_uint64),
        ('tcpDiscardedAsSender', c_uint64),
        ('tcpDiscardedAsReceiver', c_uint64),
        ('givenUpAsSender', c_uint64),
        ('givenUpAsReceiver', c_uint64),
        ('givenUpAsDropper', c_uint64),

        ('maxDelayAsSender', c_double),
        ('maxDelayAsReceiver', c_double),
        ('avgDelayAsSender', c_double),
        ('avgDelayAsReceiver', c_double),
        ('avgJitterAsSender', c_double),
        ('avgJitterAsReceiver', c_double),
        ('avgTcpSenderSrttAsSender', c_double),
        ('avgTcpSenderSrttAsReceiver', c_double),

        ('udpRemainingFlowsAsSender', c_uint32),
        ('udpRemainingFlowsAsReceiver', c_uint32),
        ('udpRemainingFlowBytesAsSender', c_uint64),
        ('udpRemainingFlowBytesAsReceiver', c_uint64),
        ('udpRemainingFlowDatarateSumAsSender', c_uint64),
        ('udpRemainingFlowDatarateSumAsReceiver', c_uint64),
        ('tcpRemainingFlowsAsSender', c_uint32),
        ('tcpRemainingFlowsAsReceiver', c_uint32),
        ('tcpRemainingFlowBytesAsSender', c_uint64),
        ('tcpRemainingFlowBytesAsReceiver', c_uint64),
        ('tcpRemainingFlowCWndSumAsSender', c_uint64),
        ('tcpRemainingFlowCWndSumAsReceiver', c_uint64),

        ('transmittedAsEdgeSrc', c_uint64),
        ('transmittedAsEdgeDst', c_uint64),
        ('transmittableAsEdgeSrc', c_uint64),
        ('transmittableAsEdgeDst', c_uint64),
        ('macTxDroppedAsEdgeSrc', c_uint64),
        ('macTxDroppedAsEdgeDst', c_uint64),
        ('phyRxDroppedAsEdgeSrc', c_uint64),
        ('phyRxDroppedAsEdgeDst', c_uint64),

        ('dropRatioAsSender', c_double),
        ('dropRatioAsReceiver', c_double),
        ('inFlightRatioAsSender', c_double),
        ('inFlightRatioAsReceiver', c_double),

        ('maxLinkUtilizationAsEdgeSrc', c_double),
        ('maxLinkUtilizationAsEdgeDst', c_double),
        ('avgLinkUtilizationAsEdgeSrc', c_double),
        ('avgLinkUtilizationAsEdgeDst', c_double),
        ('maxTxQueueRelativeLoadAsEdgeSrc', c_double),
        ('maxTxQueueRelativeLoadAsEdgeDst', c_double),
        ('avgTxQueueRelativeLoadAsEdgeSrc', c_double),
        ('avgTxQueueRelativeLoadAsEdgeDst', c_double),

    ]
    _defaults_ = {}


observed_node_dtype = struct_to_numpy_dtype(ObservedNode)
node_meta_idx = [observed_node_dtype.names.index('observerId'), observed_node_dtype.names.index('nodeId')]


def get_accepted_node_feat_idx(acceptable_features: List[str]) -> List[int]:
    """
    Returns the indices of the accepted features in the observed node struct.
    :param acceptable_features: the accepted features
    :return: the indices of the accepted features
    """
    return sorted([observed_node_dtype.names.index(f) for f in acceptable_features])


class ObservedEdge(BaseStructure):
    """
    Models a snapshot of a (directed) observed edge with respect to a specific observer,
    which is taken with respect
    to a certain amount of passed simulation time. Such snapshots hold network performance
    statistics within the given time frame, in this case specific to an outgoing
    net device sending data over its incident channel.
    """
    _fields_ = [
        ('observerId', c_long),
        ('src', c_uint32),
        ('dst', c_uint32),
        ('isLinkUp', c_bool),

        ('channelDataRate', c_uint64),
        ('channelDelay', c_uint32),
        ('txQueueCapacity', c_uint32),
        ('txQueueLoad', c_uint32),

        ('transmitted', c_uint64),
        ('transmittable', c_uint64),
        ('macTxDropped', c_uint64),
        ('phyRxDropped', c_uint64),

        ('linkUtilization', c_double),
        ('txQueueRelativeLoad', c_double),
    ]
    _defaults_ = {}


observed_edge_dtype = struct_to_numpy_dtype(ObservedEdge)
edge_meta_idx = [observed_edge_dtype.names.index('observerId'),
                 observed_edge_dtype.names.index('src'),
                 observed_edge_dtype.names.index('dst'),
                 observed_edge_dtype.names.index('isLinkUp')]


def get_accepted_edge_feat_idx(acceptable_features: List[str]) -> List[int]:
    """
    Returns the indices of the accepted features in the observed edge struct.
    :param acceptable_features: the accepted features
    :return: the indices of the accepted features
    """
    return sorted([observed_edge_dtype.names.index(f) for f in acceptable_features])


class ObservedTrafficEntry(BaseStructure):
    """
    A traffic matrix entry that models the amount of traffic that has been sent between two nodes.
    """
    _fields_ = [
        ('tx', c_uint64),
    ]
    _defaults_ = {}


class ObservedReward(BaseStructure):
    """
    Models an observed reward from an observer's perspective.
    """
    _fields_ = [
        ('observerId', c_long),
        ('reward', c_double),
    ]
    _defaults_ = {}

# Define the numpy data types for the observed reward (used for conversion between C++ and Python)
observed_reward_dtype = struct_to_numpy_dtype(ObservedReward)
reward_meta_idx = [observed_reward_dtype.names.index('observerId')]


class ObservationInfo(BaseStructure):
    """
    Contains auxiliary information about the current observation, such as the monitoring time, elapsed time,
    and the number of nodes and edges in the network.
    """
    _fields_ = [
        ('monitoringTime', c_double),
        ('elapsedTime', c_double),

        ('numNodes', c_uint32),
        ('numEdges', c_uint32),
    ]
    _defaults_ = {}


class PackerlEnvStruct(BaseStructure):
    """
    Models the shared memory object for the 'environment'.
    The environment consists of all simulation-related information such as the topology, the current
    traffic flow arrivals, the current observation, and some control signals to steer the simulation/learning
    interaction.
    """
    _fields_ = [

        # Topology
        ('numTopologyNodes', c_uint32),
        ('topologyEdges', TopologyEdge * MAX_ELEMS),
        ('numTopologyEdgesAvailable', c_uint32),
        ('topologyAvailable', c_bool),
        ('allTopologyEdgesSent', c_bool),

        # Reward Weights
        ('globalRewardWeights', GlobalRewardWeights),
        ('globalRewardWeightsAvailable', c_bool),

        # Upcoming Events (Traffic, Link Failure etc.)
        ('upcomingEvents', Event * MAX_ELEMS),
        ('upcomingEventTypes', c_uint32 * MAX_ELEMS),
        ('numUpcomingEventsAvailable', c_uint32),
        ('upcomingEventsAvailable', c_bool),
        ('allUpcomingEventsSent', c_bool),

        # Communication delays
        ('communicationDelays', CommunicationDelay * MAX_ELEMS),
        ('numCommunicationDelaysAvailable', c_uint32),
        ('centralNode', c_uint32),
        ('communicationDelaysAvailable', c_bool),
        ('allCommunicationDelaysSent', c_bool),

        # Observation/State
        ('observedGlobals', ObservedGlobal * MAX_ELEMS),
        ('numAvailableObservedGlobals', c_uint32),
        ('allObservedGlobalsSent', c_bool),

        ('observedNodes', ObservedNode * MAX_ELEMS),
        ('numAvailableObservedNodes', c_uint32),
        ('allObservedNodesSent', c_bool),

        ('observedEdges', ObservedEdge * MAX_ELEMS),
        ('numAvailableObservedEdges', c_uint32),
        ('allObservedEdgesSent', c_bool),

        ('observedRewards', ObservedReward * MAX_ELEMS),
        ('numAvailableObservedRewards', c_uint32),
        ('allObservedRewardsSent', c_bool),

        ('observedTrafficEntries', ObservedTrafficEntry * MAX_ELEMS),
        ('numAvailableObservedTrafficEntries', c_uint32),
        ('allObservedTrafficEntriesSent', c_bool),

        ('observedInfo', ObservationInfo),

        ('observationAvailable', c_bool),

        # Control Signals
        ('simReady', c_bool),
        ('done', c_bool),
        ('doneAvailable', c_bool),
    ]

    _defaults_ = {
        'topologyAvailable': False,
        'allTopologyEdgesSent': False,

        'globalRewardWeightsAvailable': False,

        'upcomingEventsAvailable': False,
        'allUpcomingEventsSent': False,

        'communicationDelaysAvailable': False,
        'allCommunicationDelaysSent': False,

        'allObservedGlobalsSent': False,
        'allObservedNodesSent': False,
        'allObservedEdgesSent': False,
        'allObservedTrafficEntriesSent': False,
        'allObservedRewardsSent': False,
        'observationAvailable': False,

        'simReady': False,
        'done': False,
        'doneAvailable': False,
    }


# ====================================================================


class RoutingActionComponent(BaseStructure):
    """
    Models a routing action component, which is a tuple of 4 values:
    - the source node of the edge
    - the destination node of the edge
    - the destination of the packet
    - the value of the routing action
    """
    _fields_ = [
        ('edgeSrc', c_uint32),
        ('edgeDst', c_uint32),
        ('trafficDst', c_uint32),
        ('value', c_float),
    ]
    _defaults_ = {}

routing_action_dtype = struct_to_numpy_dtype(RoutingActionComponent)


class RoutingAgentDelay(BaseStructure):
    """
    TODO desc
    """
    _fields_ = [
        ('nodeId', c_uint32),
        ('delay', c_float),
    ]
    _defaults_ = {}

routing_action_node_delay_dtype = struct_to_numpy_dtype(RoutingAgentDelay)


class PackerlActStruct(BaseStructure):
    """
    Models the shared memory object for the 'action space'.
    It consists of the currently selected action of the model as well as
    a logical flow control variable. Each action consists of providing
    routing descriptors for all nodes.
    """
    _fields_ = [
        ('routingActions', RoutingActionComponent * MAX_ELEMS),
        ('numRoutingActionsAvailable', c_uint32),
        ('actionsAvailable', c_bool),
        ('allActionsSent', c_bool),
        ('routingAgentDelays', RoutingAgentDelay * MAX_ELEMS),
        ('numRoutingAgentDelaysAvailable', c_uint32),
        ('routingAgentDelaysAvailable', c_bool),
        ('allRoutingAgentDelaysSent', c_bool),
    ]
    _defaults_ = {
        'actionsAvailable': False,
        'allActionsSent': False,
        'routingAgentDelaysAvailable': False,
        'allRoutingAgentDelaysSent': False,
    }


# ====================================================================

# A dict containing all memory block sizes (for convenient access)
ALL_SHM_SIZES = {
    'ENV': sizeof(PackerlEnvStruct),
    'ACT': sizeof(PackerlActStruct),
    'TOTAL': sizeof(PackerlEnvStruct) + sizeof(PackerlActStruct),
}
