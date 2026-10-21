/*
 * These structs define basic components of the entire simulation framework
 * such as network graphs and their parts, monitoring structures and
 * action spaces. As we're using ns3-ai and its shared memory paradigm, we have to
 * define all those data structures that are accessed both from the simulator
 * side (C++) as well as the learning/control side (python) as structs on both
 * sides and even take care of the member ordering to guarantee identical struct
 * sizes.
 *
 * WARNING: ALWAYS MAKE SURE THE ORDER OF MEMBERS IN THE STRUCTS IS IDENTICAL ON BOTH SIDES!
 *        OTHERWISE, THE SHARED MEMORY WILL BE MISALIGNED AND READ VALUES WILL BE ARBITRARY DESPITE MATCHING NAMES!
 */

#ifndef NS3_PACKERL_SHARED_STRUCTS_H
#define NS3_PACKERL_SHARED_STRUCTS_H

#include "ns3/ptr.h"

#include <stdint.h>
#include <vector>
#include <tuple>

using std::vector, std::tuple;

/**
 * We can send arbitrary amounts of data between the C++ and Python side, but we have to
 * define a maximum number of elements for each data structure because we have to know the
 * size of the memory block to allocate in advance. If there is more data to send than
 * MAX_ELEMS allows us to, we split the data into multiple blocks and send them sequentially.
 */
const uint32_t MAX_ELEMS = 64000;

/**
 * Models a network edge, which is a point-to-point communication channel characterized by
 * bandwidth, 2 nodes, as well as channel delay.
 */
typedef struct TopologyEdge
{
    uint32_t fst;  // first node ID
    uint32_t snd;  // second node ID
    uint64_t datarate;  // in bps
    uint32_t delay;  // in ms

} TopologyEdge;


/**
 * Models a traffic flow arrival event, which is characterized by a source and destination node,
 * an amount of data to send, a datarate, and a flag indicating whether the traffic is TCP or not.
 * In the 'upcomingEventTypes' array, this event is represented by the value 0.
 */
typedef struct TrafficFlowArrival
{
    double t;  // time since the start of the simulation when the flow arrival is created
    uint32_t src;  // source node ID
    uint32_t dst;  // destination node ID
    uint64_t amount;  // in bytes
    uint64_t datarate;  // in bps. Zero for TCP traffic since TCP senders will adapt their sending rate themselves.
    bool isTCP;  // true if the flow is TCP, false otherwise

} TrafficFlowArrival;


/**
 * Models a link failure event, which is characterized by a timestamp and the two nodes that are
 * connected by the failed link. In the 'upcomingEventTypes' array, this event is represented by the value 1.
 */
typedef struct LinkFailure
{
    double t;  // time since the start of the simulation when the link failure occurs
    uint32_t fst;  // first node ID
    uint32_t snd;  // second node ID

} LinkFailure;


/**
 * Models an event that can occur in the simulation. This is a union type that can hold
 * either a TrafficFlowArrival or a LinkFailure event. The type of the event is determined by
 * the 'upcomingEventTypes' array.
 */
union Event {
    TrafficFlowArrival trafficFlowArrival;
    LinkFailure linkFailure;
};

/**
 * Models a communication delay between two nodes in the network, which is characterized by
 * the source and destination node IDs as well as the delay in milliseconds.
 */
typedef struct CommunicationDelay
{
    uint32_t src;
    uint32_t dst;

    double delay;

} CommunicationDelay;


typedef struct GlobalRewardWeights
{
    double txWeight;
    double reTxWeight;
    double inFlightWeight;
    double rxWeight;
    double deliveredWeight;
    double ttlDroppedWeight;
    double tcpDiscardedWeight;
    double givenUpWeight;

    double maxDelayWeight;
    double avgDelayWeight;
    double avgJitterWeight;
    double avgTcpSenderSrttWeight;

    double dropRatioWeight;
    double inFlightRatioWeight;
    double maxLinkUtilizationWeight;
    double avgLinkUtilizationWeight;
    double maxTxQueueRelativeLoadWeight;
    double avgTxQueueRelativeLoadWeight;

    double transmittedWeight;
    double macTxDroppedWeight;
    double phyRxDroppedWeight;

} GlobalRewardWeights;


typedef struct ObservedGlobal
{
    long int observerId;  // -1 for global

    uint64_t tx;
    uint64_t reTx;
    uint64_t inFlight;
    uint64_t rx;
    uint64_t delivered;
    uint64_t ttlDropped;
    uint64_t tcpDiscarded;
    uint64_t givenUp;  // dropped + TCP-discarded + lost

    double maxDelay;
    uint32_t delayCount;  // used to calculate ep.-wise global avg.
    double avgDelay;
    uint32_t jitterCount;  // used to calculate ep.-wise global avg.
    double avgJitter;
    uint32_t tcpSenderSrttCount;  // used to calculate ep.-wise global avg.
    double avgTcpSenderSrtt;

    uint32_t udpRemainingFlows;
    uint64_t udpRemainingFlowBytes;
    uint64_t udpRemainingFlowDatarateSum;
    uint32_t tcpRemainingFlows;
    uint64_t tcpRemainingFlowBytes;
    uint64_t tcpRemainingFlowCWndSum;

    uint64_t txQueueCapacity;  // total capacity of all tx queues (bytes)
    uint64_t txQueueLoad;  // total load of all tx queues (bytes)

    uint64_t transmitted;  // incl. packet headers and overhead
    uint64_t transmittable;  // max. bytes sendable in cur. time frame
    uint64_t macTxDropped;
    uint64_t phyRxDropped;

    double dropRatio;  // given-up pkts. / given-up + delivered pkts.
    double inFlightRatio;  // in-flight pkts. / throughput
    double maxLinkUtilization;  // max. LU over all links
    double avgLinkUtilization;  // avg. LU over all links
    double maxTxQueueRelativeLoad;  // over all links
    double avgTxQueueRelativeLoad;  // over all links

} ObservedGlobal;


/**
 * Models a snapshot of a monitoring node, which is taken with respect
 * to a certain amount of passed simulation time. 
 * Such snapshots hold network performance
 * statistics within the given time frame, in this case specific to a node.
 */
typedef struct ObservedNode
{
    long int observerId;  // -1 for global
    uint32_t nodeId;

    uint64_t txAsSender;
    uint64_t txAsReceiver;
    uint64_t reTxAsSender;
    uint64_t reTxAsReceiver;
    uint64_t inFlightAsSender;
    uint64_t inFlightAsReceiver;
    uint64_t rxAsSender;
    uint64_t rxAsReceiver;
    uint64_t deliveredAsSender;
    uint64_t deliveredAsReceiver;
    uint64_t ttlDroppedAsSender;
    uint64_t ttlDroppedAsReceiver;
    uint64_t ttlDroppedAsDropper;
    uint64_t tcpDiscardedAsSender;
    uint64_t tcpDiscardedAsReceiver;
    uint64_t givenUpAsSender;  // incl. dropped and TCP-discarded
    uint64_t givenUpAsReceiver;  // incl. dropped and TCP-discarded
    uint64_t givenUpAsDropper;  // incl. dropped and TCP-discarded

    double maxDelayAsSender;
    double maxDelayAsReceiver;
    double avgDelayAsSender;
    double avgDelayAsReceiver;
    double avgJitterAsSender;
    double avgJitterAsReceiver;
    double avgTcpSenderSrttAsSender;
    double avgTcpSenderSrttAsReceiver;

    uint32_t udpRemainingFlowsAsSender;
    uint32_t udpRemainingFlowsAsReceiver;
    uint64_t udpRemainingFlowBytesAsSender;
    uint64_t udpRemainingFlowBytesAsReceiver;
    uint64_t udpRemainingFlowDatarateSumAsSender;
    uint64_t udpRemainingFlowDatarateSumAsReceiver;
    uint32_t tcpRemainingFlowsAsSender;
    uint32_t tcpRemainingFlowsAsReceiver;
    uint64_t tcpRemainingFlowBytesAsSender;
    uint64_t tcpRemainingFlowBytesAsReceiver;
    uint64_t tcpRemainingFlowCWndSumAsSender;
    uint64_t tcpRemainingFlowCWndSumAsReceiver;

    uint64_t transmittedAsEdgeSrc;  // incl. all headers and overhead
    uint64_t transmittedAsEdgeDst;  // incl. all headers and overhead
    uint64_t transmittableAsEdgeSrc;  // max. bytes sendable in cur. time frame
    uint64_t transmittableAsEdgeDst;  // max. bytes sendable in cur. time frame
    uint64_t macTxDroppedAsEdgeSrc;
    uint64_t macTxDroppedAsEdgeDst;
    uint64_t phyRxDroppedAsEdgeSrc;
    uint64_t phyRxDroppedAsEdgeDst;

    double dropRatioAsSender;  // given-up pkts. / given-up + delivered pkts.
    double dropRatioAsReceiver;  // given-up pkts. / given-up + delivered pkts.
    double inFlightRatioAsSender;  // in-flight pkts. / throughput
    double inFlightRatioAsReceiver;  // in-flight pkts. / throughput

    double maxLinkUtilizationAsEdgeSrc;  // over all links starting at this node
    double maxLinkUtilizationAsEdgeDst;  // over all links leading to this node
    double avgLinkUtilizationAsEdgeSrc;  // over all links starting at this node
    double avgLinkUtilizationAsEdgeDst;  // over all links leading to this node
    double maxTxQueueRelativeLoadAsEdgeSrc;  // over all links starting at this node
    double maxTxQueueRelativeLoadAsEdgeDst;  // over all links leading to this node
    double avgTxQueueRelativeLoadAsEdgeSrc;  // over all links starting at this node
    double avgTxQueueRelativeLoadAsEdgeDst;  // over all links leading to this node

} ObservedNode;


/**
 * Models a snapshot of a (directed) monitoring edge, which is taken with respect
 * to a certain amount of passed simulation time. 
 * Such snapshots hold network performance
 * statistics within the given time frame, in this case specific to an outgoing
 * net device sending data over its incident channel.
 */
typedef struct ObservedEdge
{
    long int observerId;  // -1 for global/non-delay
    uint32_t src;
    uint32_t dst;
    bool isLinkUp;

    uint64_t channelDataRate;
    uint32_t channelDelay;
    uint32_t txQueueCapacity;
    uint32_t txQueueLoad;

    uint64_t transmitted;
    uint64_t transmittable;
    uint64_t macTxDropped;
    uint64_t phyRxDropped;

    double linkUtilization;
    double txQueueRelativeLoad;

} ObservedEdge;


/**
 * A traffic matrix entry that models the amount of traffic that has been sent between two nodes.
 */
typedef struct ObservedTrafficEntry
{
    uint64_t tx;
} ObservedTrafficEntry;


/**
 * Models a reward observation, which is a tuple of two values:
 * - the observer ID (which is -1 for global)
 * - the reward value
 */
typedef struct ObservedReward
{
    long int observerId;  // -1 for global
    double reward;

} ObservedReward;


/**
 * Models the information about the current monitoring state, which is a tuple of two values:
 * - the current simulation time in seconds
 * - the time since the last monitoring in seconds
 */
typedef struct ObservationInfo
{
    double monitoringTime;
    double elapsedTime;

    uint32_t numNodes;
    uint32_t numEdges;
} ObservationInfo;


/**
 * Models a network observation, which is a tuple of 5 vectors:
 * - global observations (one actual global observation and V node-centric global observations)
 * - node observations (V node observations for V nodes, once from global view and V times from node-centric views)
 * - edge observations (E edge observations for E edges, once from global view and V times from node-centric views)
 * - matrix entry observations (V*V matrix entry observations for src-dst pairs, once from global view and V times from node-centric views)
 * - reward observations (one global reward and V node-centric rewards)
 * - the observation info (current simulation time and time since last monitoring)
 */
typedef tuple<vector<ObservedGlobal>,
              vector<ObservedNode>,
              vector<ObservedEdge>,
              vector<ObservedTrafficEntry>,
              vector<ObservedReward>,
              ObservationInfo> NetworkObservation;

/**
 * Models the shared memory object for the 'environment'.
 * The environment consists of all simulation-related information such as the topology, the current
 * traffic flows, the current monitoring, and some control signals to steer the simulation/learning
 * interaction.
 */
typedef struct PackerlEnvStruct
{
    // --- Topology ---

    uint32_t numTopologyNodes;  // number of nodes in the network
    TopologyEdge topologyEdges[MAX_ELEMS];  // edges (with their parameters) in the network
    uint32_t numTopologyEdgesAvailable;  // number of edges that are currently communicated
    bool topologyAvailable = false;  // true if the topology is available in the shared memory, false otherwise
    bool allTopologyEdgesSent = false;  // true if all topology edges have been sent, false otherwise

    // --- Reward Weights ---

    GlobalRewardWeights globalRewardWeights;  // weights for the global reward function
    bool globalRewardWeightsAvailable = false;  // true if global reward weights are available in the shared memory

    // --- Upcoming Events (Traffic, Link Failure etc.) ---

    Event upcomingEvents[MAX_ELEMS];  // upcoming events in the simulation
    uint32_t upcomingEventTypes[MAX_ELEMS];  // types of upcoming events in the simulation
    uint32_t numUpcomingEventsAvailable;  // number of upcoming events that are currently communicated
    bool upcomingEventsAvailable = false;  // true if upcoming events are available in the shared memory, false otherwise
    bool allUpcomingEventsSent = false;  // true if all upcoming events have been sent, false otherwise

    // --- Communication Delays ---

    CommunicationDelay communicationDelays[MAX_ELEMS];  // communication delays for all node pairs in the network
    uint32_t numCommunicationDelaysAvailable;  // number of communication delays that are currently communicated
    uint32_t centralNode;  // the central node in the network, i.e. the node with the lowest maximum delay to all other nodes
    bool communicationDelaysAvailable = false;  // true if communication delays are available in the shared memory, false otherwise
    bool allCommunicationDelaysSent = false;  // true if all communication delays have been sent, false otherwise

    // --- Observation/State ---

    ObservedGlobal observedGlobals[MAX_ELEMS];  // global observations, there will be V+1 of these ('+1' for global obs)
    uint32_t numAvailableObservedGlobals;
    bool allObservedGlobalsSent = false;

    ObservedNode observedNodes[MAX_ELEMS];  // there will be (V+1) * V of these ('+1' for global obs)
    uint32_t numAvailableObservedNodes;
    bool allObservedNodesSent = false;

    ObservedEdge observedEdges[MAX_ELEMS];  // there will be (V+1) * E of these ('+1' for global obs)
    uint32_t numAvailableObservedEdges;
    bool allObservedEdgesSent = false;

    ObservedReward observedRewards[MAX_ELEMS];  // there will be V+1 of these ('+1' for global reward)
    uint32_t numAvailableObservedRewards;
    bool allObservedRewardsSent = false;

    ObservedTrafficEntry observedTrafficEntries[MAX_ELEMS];  // there will be V*V of these
    uint32_t numAvailableObservedTrafficEntries;
    bool allObservedTrafficEntriesSent = false;

    ObservationInfo observedInfo;  // current simulation time and time since last monitoring

    bool observationAvailable = false;

    // --- Control Signals ---

    bool simReady = false;  // set to true by the sim when the simulation is ready to be used by the RL learning loop
    bool done = false;  // set to false by the RL learning loop at the start. If set to true by the sim, the current env is terminated
    bool doneAvailable = false;

} PackerlEnvStruct;


/**
 * Models a routing action component, which is a tuple of 4 values:
 * - the source node of the edge
 * - the destination node of the edge
 * - the destination of the packet
 * - the value of the routing action
 */
typedef struct RoutingActionComponent
{
    uint32_t edgeSrc;
    uint32_t edgeDst;
    uint32_t trafficDst;
    float value;

} RoutingActionComponent;


/**
 * Models a routing agent delay, which is a tuple of 2 values:
 * - the node ID of the routing agent
 * - the delay value (i.e. by how much time the routing agent's action shall be delayed)
 */
typedef struct RoutingAgentDelay
{
    uint32_t nodeId;
    float delay;

} RoutingAgentDelay;


/**
 * Models the shared memory object for the 'action space'.
 * It consists of the currently selected action of the model as well as
 * a logical flow control variable. Each action consists of providing
 * routing descriptors for all nodes.
 */
typedef struct PackerlActStruct
{
    RoutingActionComponent routingActions [MAX_ELEMS];  // routing actions for all nodes in the network
    uint32_t numRoutingActionsAvailable;  // number of routing actions that are currently communicated
    bool actionsAvailable = false;  // true if routing actions are available in the shared memory, false otherwise
    bool allActionsSent = false;  // true if all routing actions have been sent, false otherwise
    RoutingAgentDelay routingAgentDelays [MAX_ELEMS];  // routing agent delays for all nodes in the network
    uint32_t numRoutingAgentDelaysAvailable;  // number of routing agent delays that are currently communicated
    bool routingAgentDelaysAvailable = false;  // true if routing agent delays are available in the shared memory, false otherwise
    bool allRoutingAgentDelaysSent = false;  // true if all routing agent delays have been sent, false otherwise

} PackerlActStruct;

#endif /*NS3_PACKERL_SHARED_STRUCTS_H*/