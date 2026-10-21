#ifndef NS3_PACKERL_MONITOR_H
#define NS3_PACKERL_MONITOR_H

#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/ipv4-flow-probe-tag.h"
#include "ns3/node.h"
#include "ns3/ptr.h"
#include "ns3/address.h"
#include "ns3/ipv4-address.h"

#include "../applications/disposable-bulk-send-application.h"
#include "../applications/disposable-onoff-application.h"
#include "../odd-routing/odd-label-map.h"
#include "../utils/types.h"
#include "../utils/shared-structs.h"

#include "global-rewarder.h"
#include "monitoring-node.h"
#include "monitoring-edge.h"
#include "node-rewarder.h"

#include <memory>

using std::vector, std::unordered_map, std::pair, std::tuple;

typedef tuple<U32ToU32ToDMap, U32ToU32ToDMap, U32ToU32ToU32Map,
              U32ToU32ToDMap, U32ToU32ToU32Map> DelayAndJitter;
typedef tuple<U32ToU32ToDMap, U32ToU32ToU32Map,
              U32ToU32ToU32Map, U32ToU32ToU64Map, U32ToU32ToU64Map,
              U32ToU32ToU32Map, U32ToU32ToU64Map, U32ToU32ToU64Map> ApplicationInfo;

namespace ns3
{

class PackerlMonitor : public Object
{
    public:
        static TypeId GetTypeId(void);
        PackerlMonitor();

        // Setters for various parameters
        void SetRoutingLabelMap(Ptr<OddLabelMap> labelMap) {m_routingLabelMap = labelMap;};
        void SetCapLinkUtilization(bool capLinkUtilization) {m_capLinkUtilization = capLinkUtilization;};
        void SetRewardParameters(double localRewardSpatialDecay, uint32_t localRewardPathCutoff,
                                 bool localRewardCapToBytes);
        void SetGlobalRewardWeights(const GlobalRewardWeights& weights) {m_globalRewarder->SetWeights(weights);};
        void SetCommunicationDelaysAndCentralNode(const U32ToU32ToDMap& delays, uint32_t centralNode);
        void SetBirdseyeObserverId(int observerId) {m_birdseyeObserverId = observerId;};

        // === Registration ============================================================================================

        /**
         * Register given node with the monitor,
         * hooking traces that concern it to the monitor and node rewarder objects.
         */
        void RegisterNode(Ptr<Node> node, Ptr<PacketSink> udpSinkApp, Ptr<PacketSink> tcpSinkApp);

        /**
         * Register a monitoring edge for the given PointToPointNetDevice pair.
         */
        void RegisterEdges(Ptr<PointToPointNetDevice> firstDev, pair<uint32_t, uint32_t> firstND,
                           Ptr<PointToPointNetDevice> secondDev, pair<uint32_t, uint32_t> secondND,
                           uint64_t datarate, uint32_t delay, uint32_t txQueueCapacity);

        /**
         * Register a TCP sender application with the monitor, connecting the application's traces to the monitor.
         */
        void RegisterTcpSenderApp(Ptr<Application> app);

        /**
         * Register a UDP sender application with the monitor, connecting the application's traces to the monitor.
         */
        void RegisterUdpSenderApp(Ptr<Application> app);

        /**
         * Process a link failure event between two nodes fst and snd.
         */
        void RegisterLinkDown(uint32_t fst, uint32_t snd);

        // === Observation =============================================================================================

        /**
         * Obtain monitoring snapshots for nodes, edges, and global views, and propagate these to create node-specific
         * views of the network state ("observations").
         */
        NetworkObservation ObserveNetwork();

    private:

        // === Structs =================================================================================================

        typedef struct PacketInfo
        {
            uint32_t flowId;
            uint32_t packetId;
            uint32_t srcNodeId;
            uint32_t dstNodeId;
            uint32_t appDataSize;
        } PacketInfo;

        /**
         * Describes an edge snapshot as viewed from the source node.
         */
        typedef struct OutgoingEdgeSnapshot
        {
            uint64_t channelDataRate;
            uint32_t channelDelay;

            uint32_t txQueueLoad;
            uint32_t txQueueCapacity;

            uint64_t transmitted;
            uint64_t transmittable;
            uint64_t macTxDropped;

            bool isLinkUp;

        } OutgoingEdgeSnapshot;

        typedef unordered_map<uint32_t, OutgoingEdgeSnapshot> OutgoingEdgeSnapshotMap;

        /**
         * Describes an edge snapshot as viewed from the destination node.
         */
        typedef struct IncomingEdgeSnapshot
        {
            uint64_t phyRxDropped;
            bool isLinkUp;
        } IncomingEdgeSnapshot;

        typedef unordered_map<uint32_t, IncomingEdgeSnapshot> IncomingEdgeSnapshotMap;

        /**
         * A snapshot of a node's state at a given point in time.
         */
        typedef struct NodeSnapshot
        {
            uint32_t nodeId;

            U32ToU64Map txPerReceiver;
            U32ToU64Map reTxPerReceiver;
            U32ToU64Map rxPerSender;
            U32ToU64Map deliveredPerSender;
            U32PairToU64Map ttlDroppedPerPair;  // as dropping node
            U32ToU64Map tcpDiscardedPerSender;

            U32ToU64Map cumulTxPerReceiver;
            U32ToU64Map cumulDeliveredPerSender;
            U32PairToU64Map cumulTtlDroppedPerPair;  // as dropping node
            U32ToU64Map cumulTcpDiscardedPerSender;

            U32ToDMap maxDelayPerSender;
            U32ToDMap delaySumPerSender;
            U32ToU32Map delayCountPerSender;
            U32ToDMap jitterSumPerSender;
            U32ToU32Map jitterCountPerSender;

            U32ToDMap tcpSenderSrttSumPerReceiver;
            U32ToU32Map tcpSenderSrttCountPerReceiver;

            U32ToU32Map udpRemainingFlowsPerReceiver;
            U32ToU64Map udpRemainingBytesPerReceiver;
            U32ToU64Map udpDatarateSumPerReceiver;
            U32ToU32Map tcpRemainingFlowsPerReceiver;
            U32ToU64Map tcpRemainingBytesPerReceiver;
            U32ToU64Map tcpCWndSumPerReceiver;

            OutgoingEdgeSnapshotMap outgoingEdgeSnapshots;
            IncomingEdgeSnapshotMap incomingEdgeSnapshots;

        } NodeSnapshot;

        // === Registration ============================================================================================

        /**
         * Connects a network device to the monitor, hooking its traces to the provided monitoring edge.
         */
        void ConnectNetDevice(Ptr<PointToPointNetDevice> dev, Ptr<MonitoringEdge> monitoringEdge);

        // === Packet Tracking =========================================================================================

        // --- Utilities ---
        bool getPacketInfo(Ptr<const Packet> p, PacketInfo& packetInfo);
        tuple<uint32_t, uint32_t, uint32_t, vector<uint32_t>> unregisterReceivedPacket(Ptr<const Packet> p,
                                                                                       const PacketInfo& packetInfo);

        // --- App Tx ---
        void registerUdpTx(Ptr<const Packet> p);
        void registerTcpTx(Ptr<const Packet> p, const TcpHeader&,
                           const Address&, const Address&, Ptr<const TcpSocketBase>);
        void registerTcpRetransmit(Ptr<const Packet> p, const TcpHeader& header,
                                   const Address& from, const Address& to, Ptr<const TcpSocketBase>);

        // --- App Rx ---
        void registerUdpRx(Ptr<const Packet> p, const Address&);
        void registerTcpRx(Ptr<const Packet> p, const TcpHeader& header, const Ipv4FlowProbeTag& fTag,
                           const Address& from, const Address& to, Ptr<const TcpSocketBase>);
        void registerTcpRxDiscard(Ptr<const Packet> p, const TcpHeader& header,
                                  const Address& from, const Address& to, Ptr<const TcpSocketBase>);

        // --- Flow Monitor / Flow Probe (IP) ---
        void registerFlowmonSent(Ptr<Node> node, uint32_t flowId, uint32_t packetId, uint32_t packetSize,
                                 uint32_t appDataSize, Address fromAddr, Address toAddr);
        void registerFlowmonForward(Ptr<Node> node, uint32_t flowId, uint32_t packetId);
        void registerFlowmonReceived(Ptr<Node> node, uint32_t flowId, uint32_t packetId,
                                     Address fromAddr, Address toAddr);
        void registerFlowmonDrop(Ptr<Node> node, uint32_t flowId, uint32_t packetId,
                                 Address fromAddr, Address toAddr, uint32_t reasonCode);
        void registerFlowmonLost(uint32_t flowId, uint32_t packetId);

        // --- ARP ---
        void registerArpDrop(Ptr<const Packet> p);

        // --- Net Device ---
        void registerPhyTx(Ptr<const Packet> p);
        void registerMacTxDrop(Ptr<const Packet> p);
        void registerPhyTxDrop(Ptr<const Packet> p);
        void registerPhyRxDrop(Ptr<const Packet> p);

        // === Monitoring ==============================================================================================

        /**
         * Get the communication delays and jitter for each flow in the flow monitor.
         */
        void GetDelayAndJitter(U32ToU32ToDMap& maxDelayPerReceiverThenSender,
                               U32ToU32ToDMap& delaySumPerReceiverThenSender,
                               U32ToU32ToU32Map& delayCountPerReceiverThenSender,
                               U32ToU32ToDMap& jitterSumPerReceiverThenSender,
                               U32ToU32ToU32Map& jitterCountPerReceiverThenSender);
        /**
         * Get information from given TCP sender application and put it into the provided maps.
         */
        void GetTcpSenderAppInfo(uint32_t nodeId, uint32_t appIndex,
                                 Ptr<DisposableBulkSendApplication> tcpSenderApp,
                                 U32ToU32ToDMap& tcpSenderSrttSumPerSenderThenReceiver,
                                 U32ToU32ToU32Map& tcpSenderSrttCountPerSenderThenReceiver,
                                 U32ToU32ToU32Map& tcpRemainingFlowsPerSenderThenReceiver,
                                 U32ToU32ToU64Map& tcpRemainingBytesPerSenderThenReceiver,
                                 U32ToU32ToU64Map& tcpCWndSumPerSenderThenReceiver);

        /**
         * Get information from given UDP sender application and put it into the provided maps.
         */
        void GetUdpSenderAppInfo(uint32_t nodeId, uint32_t appIndex,
                                 Ptr<DisposableOnOffApplication> udpSenderApp,
                                 U32ToU32ToU32Map& udpRemainingFlowsPerSenderThenReceiver,
                                 U32ToU32ToU64Map& udpRemainingBytesPerSenderThenReceiver,
                                 U32ToU32ToU64Map& udpDatarateSumPerSenderThenReceiver);

        /**
         * Get information from all TCP and UDP sender applications and put it into the provided maps.
         */
        void GetApplicationInfo(U32ToU32ToDMap& tcpSenderSrttSumPerSenderThenReceiver,
                                U32ToU32ToU32Map& tcpSenderSrttCountPerSenderThenReceiver,
                                U32ToU32ToU32Map& udpRemainingFlowsPerSenderThenReceiver,
                                U32ToU32ToU64Map& udpRemainingBytesPerSenderThenReceiver,
                                U32ToU32ToU64Map& udpDatarateSumPerSenderThenReceiver,
                                U32ToU32ToU32Map& tcpRemainingFlowsPerSenderThenReceiver,
                                U32ToU32ToU64Map& tcpRemainingBytesPerSenderThenReceiver,
                                U32ToU32ToU64Map& tcpCWndSumPerSenderThenReceiver);

        /**
         * Assemble monitoring snapshots by collecting data from all monitored nodes and edges at the current time.
         */
        void MonitorNetwork(double elapsedTime, uint64_t monitoringTimeNanoseconds);

        /**
         * Reset the monitoring state, clearing all collected data and resetting the monitoring time.
         */
        void ResetMonitor(double monitoringTime);

        // === Observation =============================================================================================

        /**
         * Find the latest monitoring time that is less than or equal to the given time.
         */
        uint64_t FindLastMonitoringTime(double now) const;

        /**
         * Get the latest node snapshots for the global/birdseye observer, which contains the latest state of all nodes.
         */
        void GetNodeSnapshotsGlobal(unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot>& nodeSnapshotsGlobal,
                                    double observationTime);

        /**
         * Get the node snapshots for all nodes of the network from the perspective of a specific observer, which may
         * contain outdated snapshots of nodes that are further away from the observer.
         */
        void GetNodeSnapshotsPerObserver(unordered_map<uint32_t,
                                                       unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot>
                                                       >& nodeSnapshotsPerObserver,
                                         double observationTime);

        /**
         * Given a map of node snapshots for a specific observer, assemble the observation for that observer.
         */
        void AssembleObservation(long int observerId,
                                 const unordered_map<uint32_t, const NodeSnapshot>& nodeSnapshots,
                                 vector<ObservedGlobal>& observedGlobals,
                                 vector<ObservedNode>& observedNodes,
                                 vector<ObservedEdge>& observedEdges,
                                 vector<ObservedTrafficEntry>& observedTrafficEntries) const;

        /**
         * Debug function to print a node snapshot to the console.
         */
        void PrintNodeSnapshot(long int observerId, const NodeSnapshot& nodeSnap, double monitoringTime) const;

        /**
         * Debug function to print an global snapshot to the console.
         */
        void PrintObservedGlobal(const ObservedGlobal& global, double monitoringTime) const;

        // === Members =================================================================================================

        /**
         * The flow monitor object.
         */
        Ptr<FlowMonitor> m_flowMonitor;

        /**
         * The flow monitor helper object.
         */
        FlowMonitorHelper m_flowHelper;

        /**
         * Holds all currently initialized simulation nodes
         */
        NodeContainer m_simNodes;

        /**
         * A map from IP addresses to node IDs, used for routing.
         */
        Ptr<OddLabelMap> m_routingLabelMap;

        /**
         * Link utilization per step is calculated via
         * <bytes transferred> / <bytes that could have been transferred in this step>.
         * Since the latter value is the product of link datarate and elapsed time,
         * the entire result can exceed 1.0 if packets arrive very early in the time step.
         * If this is set to true, the link utilization is capped at 1.0.
         */
        bool m_capLinkUtilization = false;

        /**
         * The global rewarder object.
         */
        Ptr<GlobalRewarder> m_globalRewarder;

        /**
         * The node rewarder object.
         */
        Ptr<NodeRewarder> m_nodeRewarder;

        /**
         * A map of communication delays between nodes, indexed by (srcNodeId, dstNodeId).
         */
        U32ToU32ToDMap m_communicationDelaysPerSrcThenDst;

        /**
         * The central node ID.
         */
        uint32_t m_centralNode;

        /**
         * ID of the observer that gets all latest node snapshots (used for birdseye view).
         */
        int m_birdseyeObserverId;

        // === Packet Tracking/Monitoring Data Structures ==============================================================

        /**
         * A map of all packets in flight (sent out by the sender node and in/below its network layer)
         * and their payload sizes and paths, indexed by (flowId, packetId).
         */
        unordered_map<U32Pair, pair<uint32_t, vector<uint32_t>>, pair_hash> m_packetsInFlight;

        /**
         * A map of all packets received by the receiving node (but not yet delivered to the recv. app/discarded)
         * and their payload sizes and paths, indexed by (flowId, packetId).
         */
        unordered_map<U32Pair, pair<uint32_t, vector<uint32_t>>, pair_hash> m_packetsReceived;

        /**
         * A map from node/agent ID to MonitoringNode object.
         */
        unordered_map<uint32_t, Ptr<MonitoringNode>> m_monitoredNodes;

        /**
         * A vector of node/agent IDs that are currently monitored.
         */
        vector<uint32_t> m_monitoredNodeIds;

        /**
         * A map from node-device ID pair to monitoring edge.
         */
        unordered_map<U32Pair, Ptr<MonitoringEdge>, pair_hash> m_monitoredEdges;

        /**
         * A vector of edge IDs that are currently monitored.
         */
        vector<U32Pair> m_monitoredEdgeIds;

        /**
         * A vector of monitoring times in nanoseconds, which are used to index the snapshots.
         */
        std::vector<uint64_t> m_monitoringTimes;  // in nanoseconds, sorted ascending

        typedef pair<uint32_t, unordered_map<uint64_t, const NodeSnapshot>> NodeSnapshotPair;

        /**
         * A vector of node snapshots, indexed by node ID and sorted by time.
         */
        vector<NodeSnapshotPair> m_nodeSnapshotsByTime;  // sorted ascending by node ID; inner map in nanoseconds
};

}  // namespace ns3

#endif  // NS3_PACKERL_MONITOR_H