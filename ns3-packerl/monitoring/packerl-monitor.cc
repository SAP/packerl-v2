#include "packerl-monitor.h"

#include <utility>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <bitset>

using std::make_shared, std::make_pair, std::move, std::max, std::make_tuple;

namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("PackerlMonitor");
NS_OBJECT_ENSURE_REGISTERED (PackerlMonitor);

TypeId
PackerlMonitor::GetTypeId ()
{
    static TypeId tid = TypeId("ns3::PackerlMonitor")
                            .SetParent<Object>()
                            .AddConstructor<PackerlMonitor>();
    return tid;
}

PackerlMonitor::PackerlMonitor()
{
    m_flowMonitor = m_flowHelper.GetMonitor();

    // connect traces of expected events
    m_flowMonitor->TraceConnectWithoutContext("FirstTx", MakeCallback(&PackerlMonitor::registerFlowmonSent, this));
    m_flowMonitor->TraceConnectWithoutContext("Forward", MakeCallback(&PackerlMonitor::registerFlowmonForward, this));
    m_flowMonitor->TraceConnectWithoutContext("LastRx", MakeCallback(&PackerlMonitor::registerFlowmonReceived, this));
    m_flowMonitor->TraceConnectWithoutContext("Drop", MakeCallback(&PackerlMonitor::registerFlowmonDrop, this));

    // connect traces of UN-expected events. These shouldn't happen in our setting, so if they do, we let the sim crash.
    m_flowMonitor->TraceConnectWithoutContext("LostPacket", MakeCallback(&PackerlMonitor::registerFlowmonLost, this));

    // bookkeeping
    m_simNodes = NodeContainer();
    m_globalRewarder = Create<GlobalRewarder>();
    m_nodeRewarder = Create<NodeRewarder>();
    ResetMonitor(-1.);
}

void
PackerlMonitor::SetRewardParameters(double localRewardSpatialDecay, uint32_t localRewardPathCutoff,
                                    bool localRewardCapToBytes)
{
    NS_LOG_FUNCTION(this << localRewardSpatialDecay << localRewardPathCutoff << localRewardCapToBytes);
    m_nodeRewarder->SetSpatialDecay(localRewardSpatialDecay);
    m_nodeRewarder->SetPathCutoff(localRewardPathCutoff);
    m_nodeRewarder->SetCapToBytes(localRewardCapToBytes);
}

void
PackerlMonitor::SetCommunicationDelaysAndCentralNode(const U32ToU32ToDMap& delays, uint32_t centralNode)
{
    NS_LOG_FUNCTION(this);
    m_communicationDelaysPerSrcThenDst = delays;
    m_centralNode = centralNode;
}

// ===================== Registration ==================================================================================

void
PackerlMonitor::RegisterNode(Ptr<Node> node, Ptr<PacketSink> udpSinkApp, Ptr<PacketSink> tcpSinkApp)
{
    uint32_t nodeId = node->GetId();
    NS_LOG_FUNCTION(this << nodeId);

    // === Register the node in all required places (always make sure it doesn't already exist) ===

    // 1. Keep track of actual simulation nodes
    if (std::find(m_simNodes.Begin(), m_simNodes.End(), node) != m_simNodes.End())
    {
        NS_FATAL_ERROR("PackerlMonitor: Node " << nodeId << " already exists in m_simNodes");
    }
    m_simNodes.Add(node);

    // 2. Create and register a MonitoringNode object
    if (m_monitoredNodes.find(nodeId) != m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("PackerlMonitor: Node " << nodeId << " already exists in m_monitoredNodes");
    }
    m_monitoredNodes.insert(make_pair(nodeId, Create<MonitoringNode>(nodeId)));

    // 3. Keep track of monitored node IDs (sorted vector)
    auto it_monitored_id = std::lower_bound(m_monitoredNodeIds.begin(), m_monitoredNodeIds.end(), nodeId);
    if (it_monitored_id != m_monitoredNodeIds.end() && *it_monitored_id == nodeId)
    {
        NS_FATAL_ERROR("PackerlMonitor: Node ID " << nodeId << " already exists in m_monitoredNodeIds");
    }
    m_monitoredNodeIds.insert(it_monitored_id, nodeId);

    // 4. setup snapshots-by-time map for the new monitored node
    NodeSnapshotPair nodeSnapPair = make_pair(nodeId, unordered_map<uint64_t, const NodeSnapshot>());
    auto it_snap_pair = std::lower_bound(m_nodeSnapshotsByTime.begin(), m_nodeSnapshotsByTime.end(), nodeSnapPair,
        [](const NodeSnapshotPair& a, const NodeSnapshotPair& b) {return a.first < b.first;});
    if (it_snap_pair != m_nodeSnapshotsByTime.end() && it_snap_pair->first == nodeId)
    {
        NS_FATAL_ERROR("PackerlMonitor: Node ID " << nodeId << " already exists in m_nodeSnapshotsByTime");
    }
    m_nodeSnapshotsByTime.insert(it_snap_pair, nodeSnapPair);

    // 5. Connect traces of expected events
    udpSinkApp->TraceConnectWithoutContext("Rx", MakeCallback(&PackerlMonitor::registerUdpRx, this));
    tcpSinkApp->TraceConnectWithoutContext("TcpRx", MakeCallback(&PackerlMonitor::registerTcpRx, this));
    tcpSinkApp->TraceConnectWithoutContext("TcpRxDiscard", MakeCallback(&PackerlMonitor::registerTcpRxDiscard, this));

    // 6. Connect traces of UN-expected events. These shouldn't happen in our setting, so if they do, we let the sim crash.
    Ptr<ArpL3Protocol> arp = node->GetObject<ArpL3Protocol>();
    if (arp == nullptr)
    {
        NS_FATAL_ERROR("m_inetHelper.Install() should have installed ArpL3Protocol, but node doesn't have it...");
    }
    arp->TraceConnectWithoutContext("Drop", MakeCallback(&PackerlMonitor::registerArpDrop, this));

    // 7. Register the node with the flowMonitor
    m_flowMonitor = m_flowHelper.Install(node);

    // 8. Finally, set up node-specific rewarder
    m_nodeRewarder->registerNode(nodeId);
}

void
PackerlMonitor::ConnectNetDevice(Ptr<PointToPointNetDevice> dev, Ptr<MonitoringEdge> monitoringEdge)
{
    NS_LOG_FUNCTION(this);

    // packet and global byte tracking
    dev->TraceConnectWithoutContext("PhyTxEnd", MakeCallback(&PackerlMonitor::registerPhyTx, this));
    dev->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&PackerlMonitor::registerMacTxDrop, this));
    dev->TraceConnectWithoutContext("PhyTxDrop", MakeCallback(&PackerlMonitor::registerPhyTxDrop, this));
    dev->TraceConnectWithoutContext("PhyRxDrop", MakeCallback(&PackerlMonitor::registerPhyRxDrop, this));

    // edge-level byte tracking is done in the MonitoringEdge directly
    dev->TraceConnectWithoutContext("PhyTxEnd", MakeCallback(&MonitoringEdge::registerTransmission, &(*monitoringEdge)));
    dev->TraceConnectWithoutContext("MacTxDrop", MakeCallback(&MonitoringEdge::registerMacTxDrop, &(*monitoringEdge)));
    dev->TraceConnectWithoutContext("PhyTxDrop", MakeCallback(&MonitoringEdge::registerMacTxDrop, &(*monitoringEdge)));
    dev->TraceConnectWithoutContext("PhyRxDrop", MakeCallback(&MonitoringEdge::registerPhyRxDrop, &(*monitoringEdge)));
}

void
PackerlMonitor::RegisterEdges(Ptr<PointToPointNetDevice> firstDev, pair<uint32_t, uint32_t> firstND,
                              Ptr<PointToPointNetDevice> secondDev, pair<uint32_t, uint32_t> secondND,
                              uint64_t datarate, uint32_t delay, uint32_t txQueueCapacity)
{
    NS_LOG_FUNCTION(this << firstND.first << firstND.second << secondND.first << secondND.second);

    // === A: Edge for first direction ===
    // A.1 Create a monitored edge
    Ptr<MonitoringEdge> firstEdge = Create<MonitoringEdge>(firstND, secondND, datarate, delay, txQueueCapacity);

    // A.2 Add IDs to sorted vector m_monitoredEdgeIds
    auto firstEdgeId = make_pair(firstND.first, secondND.first);
    auto it_firstEdgeId = std::lower_bound(m_monitoredEdgeIds.begin(), m_monitoredEdgeIds.end(), firstEdgeId);
    if (it_firstEdgeId != m_monitoredEdgeIds.end() && *it_firstEdgeId == firstEdgeId)
    {
        NS_FATAL_ERROR("RegisterEdges(): Edge " << firstND.first << "-" << secondND.first
                       << " already exists in m_monitoredEdgeIds");
    }
    m_monitoredEdgeIds.insert(it_firstEdgeId, firstEdgeId);

    // A.3 Add the edge to the map of monitored edges
    if (m_monitoredEdges.find(firstEdgeId) != m_monitoredEdges.end())
    {
        NS_FATAL_ERROR("RegisterEdges(): Edge " << firstND.first << "-" << secondND.first
                       << " already exists in m_monitoredEdges");
    }
    m_monitoredEdges.insert(make_pair(firstEdgeId, firstEdge));

    // A.4 Connect the net device to the edge
    ConnectNetDevice(firstDev, firstEdge);

    // === B: Edge for opposite direction ===
    // B.1 Create a monitored edge
    Ptr<MonitoringEdge> secondEdge = Create<MonitoringEdge>(secondND, firstND, datarate, delay, txQueueCapacity);

    // B.2 Add IDs to sorted vector m_monitoredEdgeIds
    auto secondEdgeId = make_pair(secondND.first, firstND.first);
    auto it_secondEdgeId = std::lower_bound(m_monitoredEdgeIds.begin(), m_monitoredEdgeIds.end(), secondEdgeId);
    if (it_secondEdgeId != m_monitoredEdgeIds.end() && *it_secondEdgeId == secondEdgeId)
    {
        NS_FATAL_ERROR("RegisterEdges(): Edge " << secondND.first << "-" << firstND.first
                       << " already exists in m_monitoredEdgeIds");
    }
    m_monitoredEdgeIds.insert(it_secondEdgeId, secondEdgeId);

    // B.3 Add the edge to the map of monitored edges
    if (m_monitoredEdges.find(secondEdgeId) != m_monitoredEdges.end())
    {
        NS_FATAL_ERROR("RegisterEdges(): Edge " << secondND.first << "-" << firstND.first
                       << " already exists in m_monitoredEdges");
    }
    m_monitoredEdges.insert(make_pair(secondEdgeId, secondEdge));

    // B.4 Connect the net device to the edge
    ConnectNetDevice(secondDev, secondEdge);
}

void
PackerlMonitor::RegisterTcpSenderApp(Ptr<Application> app)
{
    NS_LOG_FUNCTION(this);
    // TCP is byte-based and puts data in packets as it sees fit -> use the Tcp socket's Tx trace
    // app->TraceConnectWithoutContext("TcpTx", MakeCallback(&PackerlMonitor::registerTcpTx, this)); [TODO once needed]
    // TCP Retransmissions are not covered by FlowMonitor, so connect the trace manually for TCP applications
    app->TraceConnectWithoutContext("TcpRetransmission", MakeCallback(&PackerlMonitor::registerTcpRetransmit, this));
}

void
PackerlMonitor::RegisterUdpSenderApp(Ptr<Application> app)
{
    NS_LOG_FUNCTION(this);
    // UDP respects application packets
    // app->TraceConnectWithoutContext("Tx", MakeCallback(&PackerlMonitor::registerUdpTx, this)); [TODO once needed]
}

void
PackerlMonitor::RegisterLinkDown(uint32_t fst, uint32_t snd)
{
    NS_LOG_FUNCTION(this << fst << snd);
    m_monitoredEdges[make_pair(fst, snd)]->registerLinkDown();
    m_monitoredEdges[make_pair(snd, fst)]->registerLinkDown();
}

// ===================== Packet Tracking ===============================================================================

// --- Utilities ------------------------------------------------------------

bool
PackerlMonitor::getPacketInfo(Ptr<const Packet> p, PacketInfo& packetInfo)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    Ipv4FlowProbeTag fTag;
    if (!p->FindFirstMatchingByteTag(fTag))
    {
        NS_LOG_LOGIC("getPacketInfo(): packet has no Ipv4FlowProbeTag. Returning without packet info.");
        return false;  // packet has no Ipv4FlowProbeTag
    }

    packetInfo.flowId = fTag.GetFlowId();
    packetInfo.packetId = fTag.GetPacketId();
    packetInfo.srcNodeId = m_routingLabelMap->getNodeIdForIpv4Address(fTag.GetSrcAddress());
    packetInfo.dstNodeId = m_routingLabelMap->getNodeIdForIpv4Address(fTag.GetDstAddress());
    packetInfo.appDataSize = fTag.GetAppDataSize();

    return true;
}

tuple<uint32_t, uint32_t, uint32_t, vector<uint32_t>>
PackerlMonitor::unregisterReceivedPacket(Ptr<const Packet> p, const PacketInfo& packetInfo)
{
    uint32_t flowId = packetInfo.flowId;
    uint32_t packetId = packetInfo.packetId;
    uint32_t srcNodeId = packetInfo.srcNodeId;
    uint32_t dstNodeId = packetInfo.dstNodeId;
    uint32_t appDataSize = packetInfo.appDataSize;

    NS_LOG_FUNCTION(this << flowId << packetId << srcNodeId << dstNodeId << appDataSize);
    NS_LOG_DEBUG(std::setw(39) << "unregisterReceivedPacket(): flowId=" << flowId << ", packetId=" << packetId
                << ", src=" << srcNodeId << ", dst=" << dstNodeId << ", appDataSize=" << appDataSize);
    auto it = m_packetsReceived.find(make_pair(flowId, packetId));
    if (it == m_packetsReceived.end())
    {
        NS_FATAL_ERROR("unregisterReceivedPacket(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked received packets. This shouldn't happen.");
    }

    uint32_t registeredAppDataSize = it->second.first;
    const vector<uint32_t> packetPath = it->second.second;
    if (registeredAppDataSize != appDataSize)
    {
        NS_FATAL_ERROR("unregisterReceivedPacket(): app data size of packet " << packetId << " of flow " << flowId
                       << "(" << appDataSize << ") does not match the registered app data size ("
                       << registeredAppDataSize << ")");
    }
    m_packetsReceived.erase(it);
    return make_tuple(srcNodeId, dstNodeId, appDataSize, packetPath);
}

// --- App Tx ----------------------------------------------------------------

void
PackerlMonitor::registerUdpTx(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    // CHECK/TRACK PACKET: none (happens later)

    // REWARD/PENALIZE: none

    // TRACK BYTES: in MonitoringNode
    // m_appTxGlobal += p->GetSize();
}

void
PackerlMonitor::registerTcpTx(Ptr<const Packet> p, const TcpHeader&,
                              const Address&, const Address&, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    // CHECK/TRACK PACKET: none (happens later)

    // REWARD/PENALIZE: none

    // TRACK BYTES: in MonitoringNode
    // m_appTxGlobal += p->GetSize();
}

void
PackerlMonitor::registerTcpRetransmit(Ptr<const Packet> p, const TcpHeader& header,
                                      const Address& from, const Address& to, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    // CHECK/TRACK PACKET: none (happens later)

    // REWARD/PENALIZE: none

    // TRACK BYTES: in MonitoringNode
    uint32_t srcNodeId = m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(from));
    uint32_t dstNodeId = m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(to));
    uint32_t appDataSize = p->GetSize();  // This is OK because TcpHeader has not yet been added to the packet
    auto it_node = m_monitoredNodes.find(dstNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerTcpRetransmit(): destination node " << dstNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerTcpRetransmit(srcNodeId, appDataSize);
}

// --- App Rx ----------------------------------------------------------------

void
PackerlMonitor::registerUdpRx(Ptr<const Packet> p, const Address&)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    PacketInfo packetInfo;
    if (! getPacketInfo(p, packetInfo))
    {
        NS_LOG_LOGIC("registerUdpRx(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    
    // CHECK/TRACK PACKET: unregister from received map
    auto packetStats = unregisterReceivedPacket(p, packetInfo);
    uint32_t srcNodeId = get<0>(packetStats);
    uint32_t dstNodeId = get<1>(packetStats);
    uint32_t appDataSize = get<2>(packetStats);
    const vector<uint32_t>& packetPath = get<3>(packetStats);

    // REWARD/PENALIZE: handle UDP Rx
    m_nodeRewarder->HandleAppRx(appDataSize, packetPath);

    // TRACK BYTES: in MonitoringNode
    auto it_node = m_monitoredNodes.find(dstNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerUdpRx(): destination node " << dstNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerUdpRx(srcNodeId, appDataSize);
}

void
PackerlMonitor::registerTcpRx(Ptr<const Packet> p, const TcpHeader& header, const Ipv4FlowProbeTag& fTag,
                              const Address& from, const Address& to, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    // p is the packet BEFORE put in the RxBuffer of the TCP socket, so it should still be registered.
    // However, if it's a meta-packet (e.g. SYN/ACK), it may carry no application data, meaning that the
    // Ipv4FlowProbeTag byte tag has to be supplied to be accessible.
    const PacketInfo packetInfo = {fTag.GetFlowId(), fTag.GetPacketId(),
                                   m_routingLabelMap->getNodeIdForIpv4Address(fTag.GetSrcAddress()),
                                   m_routingLabelMap->getNodeIdForIpv4Address(fTag.GetDstAddress()),
                                   fTag.GetAppDataSize()};

    // CHECK/TRACK PACKET: unregister from received map
    auto packetStats = unregisterReceivedPacket(p, packetInfo);
    uint32_t srcNodeId = get<0>(packetStats);
    uint32_t dstNodeId = get<1>(packetStats);
    uint32_t appDataSize = get<2>(packetStats);
    const vector<uint32_t>& packetPath = get<3>(packetStats);

    // REWARD/PENALIZE: handle TCP Rx
    m_nodeRewarder->HandleAppRx(appDataSize, packetPath);

    // TRACK BYTES: in MonitoringNode
    auto it_node = m_monitoredNodes.find(dstNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerTcpRx(): destination node " << dstNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerTcpRx(srcNodeId, appDataSize);
}

void
PackerlMonitor::registerTcpRxDiscard(Ptr<const Packet> p, const TcpHeader& header,
                                     const Address& from, const Address& to, Ptr<const TcpSocketBase>)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    // track packet. p is the packet BEFORE put in the RxBuffer of the TCP socket, so it should still be registered.
    // since this trace will be fired BEFORE the receiving socket will strip the TcpHeader, we can get the
    // Ipv4FlowProbeTag byte tag from the packet.
    PacketInfo packetInfo;
    if (! getPacketInfo(p, packetInfo))
    {
        NS_LOG_LOGIC("registerTcpRxDiscard(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    
    // CHECK/TRACK PACKET: unregister from received map
    auto packetStats = unregisterReceivedPacket(p, packetInfo);
    uint32_t srcNodeId = get<0>(packetStats);
    uint32_t dstNodeId = get<1>(packetStats);
    uint32_t appDataSize = get<2>(packetStats);
    const vector<uint32_t>& packetPath = get<3>(packetStats);

    // REWARD/PENALIZE: handle TCP Rx discard
    m_nodeRewarder->HandleTcpRxDiscard(appDataSize, packetPath);

    // TRACK BYTES: in MonitoringNode
    auto it_node = m_monitoredNodes.find(dstNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerTcpRxDiscard(): destination node " << dstNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerTcpRxDiscard(srcNodeId, appDataSize);
}

// --- Flowmon ----------------------------------------------------------------

void
PackerlMonitor::registerFlowmonSent(Ptr<Node> node, uint32_t flowId, uint32_t packetId, uint32_t packetSize,
                                    uint32_t appDataSize, Address fromAddr, Address toAddr)
{
    uint32_t srcNodeId = node->GetId();
    NS_LOG_FUNCTION(this << flowId << packetId << packetSize << appDataSize << srcNodeId << fromAddr << toAddr);

    uint32_t dstNodeId = m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(toAddr));

    NS_LOG_DEBUG(std::setw(39) << "registerFlowmonSent(): flowId=" << flowId << ", packetId=" << packetId
                << ", src=" << srcNodeId << ", dst=" << dstNodeId << ", appDataSize=" << appDataSize);

    // CHECK/TRACK PACKET: add to in-flight map
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it != m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerFlowmonSent(): packet " << packetId << " of flow " << flowId
                       << " already found in tracked packets in flight. This shouldn't happen.");
    }
    m_packetsInFlight.emplace(make_pair(flowId, packetId), make_pair(appDataSize, vector<uint32_t>{srcNodeId}));

    // REWARD/PENALIZE: none

    // TRACK BYTES: in MonitoringNode
    auto it_node = m_monitoredNodes.find(srcNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerFlowmonSent(): source node " << srcNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerFlowmonSent(dstNodeId, appDataSize);
}

void
PackerlMonitor::registerFlowmonForward(Ptr<Node> node, uint32_t flowId, uint32_t packetId)
{
    uint32_t forwardingNodeId = node->GetId();
    NS_LOG_FUNCTION(this << flowId << packetId << forwardingNodeId);

    NS_LOG_DEBUG(std::setw(39) << "registerFlowmonForward(): flowId=" << flowId << ", packetId=" << packetId
                << ", forwardingNodeId=" << forwardingNodeId);

    // CHECK/TRACK PACKET: append traversed node to tracked in-flight packet's path
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it == m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerFlowmonForward(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked packets in flight. This shouldn't happen.");
    }
    it->second.second.push_back(forwardingNodeId);

    // REWARD/PENALIZE: none

    // TRACK BYTES: none
}

void
PackerlMonitor::registerFlowmonReceived(Ptr<Node> node, uint32_t flowId, uint32_t packetId,
                                        Address fromAddr, Address toAddr)
{
    uint32_t dstNodeId = node->GetId();
    NS_LOG_FUNCTION(this << flowId << packetId << dstNodeId << fromAddr << toAddr);

    uint32_t srcNodeId = m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(fromAddr));

    NS_LOG_DEBUG(std::setw(39) << "registerFlowmonReceived(): flowId=" << flowId << ", packetId=" << packetId
                << ", src=" << srcNodeId << ", dst=" << dstNodeId);

    // CHECK/TRACK PACKET: move in-flight packet to received map
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it == m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerFlowmonReceived(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked packets in flight. This shouldn't happen.");
    }
    auto it_received = m_packetsReceived.find(make_pair(flowId, packetId));
    if (it_received != m_packetsReceived.end())
    {
        NS_FATAL_ERROR("registerFlowmonReceived(): packet " << packetId << " of flow " << flowId
                       << " already found in tracked received packets. This shouldn't happen.");
    }
    m_packetsReceived.emplace(move(it->first), move(it->second));
    uint32_t appDataSize = it->second.first;
    m_packetsInFlight.erase(it);

    // REWARD/PENALIZE: none (here we don't know yet if the packet will be delivered to the application!)

    // TRACK BYTES: in MonitoringNode
    auto it_node = m_monitoredNodes.find(dstNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerFlowmonReceived(): destination node " << dstNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerFlowmonReceived(srcNodeId, appDataSize);
}

void
PackerlMonitor::registerFlowmonDrop(Ptr<Node> node, uint32_t flowId, uint32_t packetId,
                                    Address fromAddr, Address toAddr, uint32_t reasonCode)
{
    uint32_t droppingNodeId = node->GetId();
    NS_LOG_FUNCTION(this << flowId << packetId << droppingNodeId << fromAddr << toAddr << reasonCode);

    // For network layer drops, we only expect and handle Ipv4FlowProbe::DropReason::DROP_TTL_EXPIRE' for now.
    // There may be other drop reasons when the drops happen elsewhere (e.g. network devices).
    // TODO can queue drops arrive here?
    if (reasonCode != Ipv4FlowProbe::DropReason::DROP_TTL_EXPIRE)
    {
        NS_FATAL_ERROR("registerFlowmonDrop(): drop reason code " << reasonCode << " unexpected.");
        return;
    }

    uint32_t srcNodeId = m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(fromAddr));
    uint32_t dstNodeId = m_routingLabelMap->getNodeIdForIpv4Address(Ipv4Address::ConvertFrom(toAddr));

    // CHECK/TRACK PACKET: unregister from in-flight map
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it == m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerFlowmonDrop(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked packets in flight. This shouldn't happen.");
    }
    uint32_t appDataSize = it->second.first;
    const vector<uint32_t> packetPath = it->second.second;
    m_packetsInFlight.erase(it);

    // REWARD/PENALIZE: handle TTL expiration drop
    m_nodeRewarder->HandleDropTTL(appDataSize, packetPath);

    // TRACK BYTES: in MonitoringNode
    auto it_node = m_monitoredNodes.find(droppingNodeId);
    if (it_node == m_monitoredNodes.end())
    {
        NS_FATAL_ERROR("registerFlowmonDrop(): dropping node " << droppingNodeId << " not found in m_monitoredNodes");
    }
    it_node->second->registerFlowmonDrop(srcNodeId, dstNodeId, appDataSize);
}

void
PackerlMonitor::registerFlowmonLost(uint32_t flowId, uint32_t packetId)
{
    NS_LOG_FUNCTION(this << flowId << packetId);
    NS_FATAL_ERROR("registerFlowmonLost(): packet " << packetId << " of flow " << flowId
                   << " lost. We don't support these for now.");

    // CHECK/TRACK PACKET: unregister from in-flight map [once supported]

    // REWARD/PENALIZE: treat as if they were dropped due to TTL expiration (i.e. taking bad paths). [once supported]

    // TRACK BYTES [once supported]
}

// --- ARP --------------------------------------------------------------------

void
PackerlMonitor::registerArpDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    NS_FATAL_ERROR("We assume ARP drops to not happen, why are we here?");
    // WARNING: At some later point, if supporting these, keep in mind there are Tx and Rx drops here

    // CHECK/TRACK PACKET: unregister from in-flight map [once supported]

    // REWARD/PENALIZE: ??? [once supported]

    // TRACK BYTES [once supported]
}

// --- Net Device ------------------------------------------------------------

void
PackerlMonitor::registerPhyTx(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    PacketInfo packetInfo;
    if (! getPacketInfo(p, packetInfo))
    {
        NS_LOG_LOGIC("registerPhyTx(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    uint32_t flowId = packetInfo.flowId;
    uint32_t packetId = packetInfo.packetId;

    // CHECK/TRACK PACKET: packet should be in-flight
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it == m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerMacTxDrop(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked packets in flight. This shouldn't happen.");
    }

    // REWARD/PENALIZE: none

    // TRACK BYTES: none (it's done in the MonitoringEdges directly)
}

void
PackerlMonitor::registerMacTxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    PacketInfo packetInfo;
    if (! getPacketInfo(p, packetInfo))
    {
        NS_LOG_LOGIC("registerMacTxDrop(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    uint32_t flowId = packetInfo.flowId;
    uint32_t packetId = packetInfo.packetId;
    uint32_t appDataSize = packetInfo.appDataSize;

    // CHECK/TRACK PACKET: unregister from in-flight map
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it == m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerMacTxDrop(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked packets in flight. This shouldn't happen.");
    }
    uint32_t storedAppDataSize = it->second.first;
    const vector<uint32_t> packetPath = it->second.second;
    if (storedAppDataSize != appDataSize)
    {
        NS_FATAL_ERROR("registerMacTxDrop(): app data size of packet " << packetId << " of flow " << flowId
                       << "(" << appDataSize << ") does not match the registered app data size ("
                       << storedAppDataSize << ")");
    }
    m_packetsInFlight.erase(it);

    // The dropping node is available to us via packetPath.back() because the causing
    // sendOutgoingTrace() or unicastForwardTrace() is fired prior to the MacTxDrop event.
    Ptr<const Node> transmitterNode = m_simNodes.Get(packetPath.back());

    // MacTxDrops usually happen when the device's TxQueue is full, but whose fault is this congestion drop?
    // If all the node's device TxQueues are full, this drop was unavoidable anyway, and we won't penalize this node.
    uint32_t packetSize = p->GetSize();  // total size incl. headers
    bool wasUnavoidable = true;
    for (uint32_t devId = 0; devId < transmitterNode->GetNDevices(); ++devId)
    {
        Ptr<PointToPointNetDevice> dev = DynamicCast<PointToPointNetDevice>(transmitterNode->GetDevice(devId));
        if (dev == nullptr) continue; // only consider PointToPointNetDevices (ignore e.g. loopback devices)

        auto devQueue = dev->GetQueue();
        if (devQueue->GetCurrentSize().GetValue() + packetSize <= devQueue->GetMaxSize().GetValue())
        {
            wasUnavoidable = false;
            break;
        }
    }

    // REWARD/PENALIZE: handle queue drop depending on whether it was unavoidable
    m_nodeRewarder->HandleDropQueue(storedAppDataSize, packetPath, wasUnavoidable);

    // TRACK BYTES: none (it's done in the MonitoringEdges directly)
}

void
PackerlMonitor::registerPhyTxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    NS_FATAL_ERROR("registerPhyTxDrop(): packet dropped at the Phy layer. "
                   << "We currently do not support those.");

    // CHECK/TRACK PACKET: unregister from in-flight map [once supported]

    // REWARD/PENALIZE: handle PhyTx drop [once supported]

    // TRACK BYTES: none (it's done in the MonitoringEdges directly) [once supported]
}

void
PackerlMonitor::registerPhyRxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());

    PacketInfo packetInfo;
    if (! getPacketInfo(p, packetInfo))
    {
        NS_LOG_LOGIC("registerPhyRxDrop(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    uint32_t flowId = packetInfo.flowId;
    uint32_t packetId = packetInfo.packetId;
    uint32_t appDataSize = packetInfo.appDataSize;

    // CHECK/TRACK PACKET: unregister from in-flight map
    auto it = m_packetsInFlight.find(make_pair(flowId, packetId));
    if (it == m_packetsInFlight.end())
    {
        NS_FATAL_ERROR("registerMacTxDrop(): packet " << packetId << " of flow " << flowId
                       << " not found in tracked packets in flight. This shouldn't happen.");
    }
    uint32_t storedAppDataSize = it->second.first;
    const vector<uint32_t> packetPath = it->second.second;
    if (storedAppDataSize != appDataSize)
    {
        NS_FATAL_ERROR("registerPhyRxDrop(): app data size of packet " << packetId << " of flow " << flowId
                       << "(" << appDataSize << ") does not match the registered app data size ("
                       << storedAppDataSize << ")");
    }
    m_packetsInFlight.erase(it);

    // REWARD/PENALIZE: handle PhyRx drop
    m_nodeRewarder->HandleDropPhyRx(storedAppDataSize, packetPath);

    // TRACK BYTES: none (it's done in the MonitoringEdges directly)
}

// ===================== Monitoring ====================================================================================

void
PackerlMonitor::GetDelayAndJitter(U32ToU32ToDMap& maxDelayPerReceiverThenSender,
                                  U32ToU32ToDMap& delaySumPerReceiverThenSender,
                                  U32ToU32ToU32Map& delayCountPerReceiverThenSender,
                                  U32ToU32ToDMap& jitterSumPerReceiverThenSender,
                                  U32ToU32ToU32Map& jitterCountPerReceiverThenSender)
{
    NS_LOG_FUNCTION(this);

    // initialize maps with empty sub-maps per node id
    for (uint32_t nodeId : m_monitoredNodeIds)
    {
        maxDelayPerReceiverThenSender[nodeId] = U32ToDMap();
        delaySumPerReceiverThenSender[nodeId] = U32ToDMap();
        delayCountPerReceiverThenSender[nodeId] = U32ToU32Map();

        jitterSumPerReceiverThenSender[nodeId] = U32ToDMap();
        jitterCountPerReceiverThenSender[nodeId] = U32ToU32Map();
    }

    // obtain packet delay and jitter values from the flowMonitor
    const FlowMonitor::FlowStatsContainer& flowMonitorStats = m_flowMonitor->GetFlowStats();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(m_flowHelper.GetClassifier());
    for (auto i = flowMonitorStats.begin(); i != flowMonitorStats.end(); ++i)
    {
        Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
        uint32_t src = m_routingLabelMap->getNodeIdForIpv4Address(t.sourceAddress);
        uint32_t dst = m_routingLabelMap->getNodeIdForIpv4Address(t.destinationAddress);
        FlowMonitor::FlowStats stats = i->second;

        if (stats.rxPackets <= 0) continue;  // skip flows with no received packets

        double maxDelay = stats.maxDelay.GetSeconds();
        U32ToDMap maxDelayPerSender = maxDelayPerReceiverThenSender.at(dst);
        if (maxDelayPerSender.find(src) == maxDelayPerSender.end() || maxDelay > maxDelayPerSender.at(src))
        {
            maxDelayPerReceiverThenSender[dst][src] = maxDelay;
        }
        delaySumPerReceiverThenSender[dst][src] += stats.delaySum.GetSeconds();
        delayCountPerReceiverThenSender[dst][src] += stats.rxPackets;

        jitterSumPerReceiverThenSender[dst][src] += stats.jitterSum.GetSeconds();
        jitterCountPerReceiverThenSender[dst][src] += stats.rxPackets - 1;
    }
}


void
PackerlMonitor::GetTcpSenderAppInfo(uint32_t nodeId, uint32_t appIndex,
                                    Ptr<DisposableBulkSendApplication> tcpSenderApp,
                                    U32ToU32ToDMap& tcpSenderSrttSumPerSenderThenReceiver,
                                    U32ToU32ToU32Map& tcpSenderSrttCountPerSenderThenReceiver,
                                    U32ToU32ToU32Map& tcpRemainingFlowsPerSenderThenReceiver,
                                    U32ToU32ToU64Map& tcpRemainingBytesPerSenderThenReceiver,
                                    U32ToU32ToU64Map& tcpCWndSumPerSenderThenReceiver)
{
    NS_LOG_FUNCTION(this);

    NS_ASSERT_MSG(tcpSenderApp->GetMaxBytes() > 0, "GetApplicationInfo(): node " << nodeId
                  << ", app: " << appIndex << " has unbounded flow size -> not supported yet");
    uint64_t remainingBytes = tcpSenderApp->GetRemainingBytes();
    if (remainingBytes <= 0)
    {
        NS_LOG_DEBUG("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                     << " has no remaining flow size");
        return;
    }
    Ptr<Socket> socket = tcpSenderApp->GetSocket();
    if (! socket)
    {
        NS_LOG_DEBUG("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                     << " has no Socket -> may be a stopped application");
        return;
    }
    Ptr<TcpSocketBase> tcpSocket = DynamicCast<TcpSocketBase>(socket);
    if (! tcpSocket)
    {
        NS_FATAL_ERROR("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                       << " active socket is not a TcpSocket");
    }
    Address peerAddress;
    bool stillConnected = tcpSocket->GetPeerName(peerAddress) == 0;
    if (! stillConnected)
    {
        NS_LOG_DEBUG("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                     << " peer Socket is not connected anymore");
        return;
    }

    // This sender app is active and has remaining flow size -> track it
    Ipv4Address peerIpv4 = InetSocketAddress::ConvertFrom(peerAddress).GetIpv4();
    uint32_t dstNodeId = m_routingLabelMap->getNodeIdForIpv4Address(peerIpv4);

    tcpRemainingFlowsPerSenderThenReceiver[nodeId][dstNodeId]++;
    tcpRemainingBytesPerSenderThenReceiver[nodeId][dstNodeId] += remainingBytes;
    tcpCWndSumPerSenderThenReceiver[nodeId][dstNodeId] += tcpSocket->GetCWnd();

    double tcpSrtt = tcpSocket->GetSrtt().GetSeconds();
    if (tcpSrtt > 0.)  // only track if an RTT estimate is available
    {
        tcpSenderSrttCountPerSenderThenReceiver[nodeId][dstNodeId]++;
        tcpSenderSrttSumPerSenderThenReceiver[nodeId][dstNodeId] += tcpSrtt;
    }
}

void
PackerlMonitor::GetUdpSenderAppInfo(uint32_t nodeId, uint32_t appIndex,
                                    Ptr<DisposableOnOffApplication> udpSenderApp,
                                    U32ToU32ToU32Map& udpRemainingFlowsPerSenderThenReceiver,
                                    U32ToU32ToU64Map& udpRemainingBytesPerSenderThenReceiver,
                                    U32ToU32ToU64Map& udpDatarateSumPerSenderThenReceiver)
{
    NS_LOG_FUNCTION(this);

    if (! udpSenderApp->IsConnected())
    {
        NS_LOG_DEBUG("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                     << " peer Socket is not connected anymore");
        return;
    }
    NS_ASSERT_MSG(udpSenderApp->GetMaxBytes() > 0, "GetApplicationInfo(): node " << nodeId
                  << ", app: " << appIndex << " has unbounded flow size -> not supported yet");
    uint64_t remainingBytes = udpSenderApp->GetRemainingBytes();
    if (remainingBytes <= 0)
    {
        NS_LOG_DEBUG("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                     << " has no remaining flow size");
        return;
    }
    Ptr<Socket> socket = udpSenderApp->GetSocket();
    if (! socket)
    {
        NS_LOG_DEBUG("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                     << " has no Socket -> may be a stopped application");
        return;
    }

    // This sender app is active and has remaining flow size -> track it
    Ptr<UdpSocket> udpSocket = DynamicCast<UdpSocket>(socket);
    if (! udpSocket)
    {
        NS_FATAL_ERROR("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                       << " active socket is not a UdpSocket");
    }
    Address peerAddress;
    bool validPeerAddress = udpSocket->GetPeerName(peerAddress) == 0;
    if (! validPeerAddress)
    {
        NS_FATAL_ERROR("GetApplicationInfo(): node " << nodeId << ", app: " << appIndex
                       << " socket does not have a valid peer address");
    }
    Ipv4Address peerIpv4 = InetSocketAddress::ConvertFrom(peerAddress).GetIpv4();
    uint32_t dstNodeId = m_routingLabelMap->getNodeIdForIpv4Address(peerIpv4);

    udpRemainingFlowsPerSenderThenReceiver[nodeId][dstNodeId]++;
    udpRemainingBytesPerSenderThenReceiver[nodeId][dstNodeId] += remainingBytes;
    udpDatarateSumPerSenderThenReceiver[nodeId][dstNodeId] += udpSenderApp->GetDataRate();
}


void
PackerlMonitor::GetApplicationInfo(U32ToU32ToDMap& tcpSenderSrttSumPerSenderThenReceiver,
                                   U32ToU32ToU32Map& tcpSenderSrttCountPerSenderThenReceiver,
                                   U32ToU32ToU32Map& udpRemainingFlowsPerSenderThenReceiver,
                                   U32ToU32ToU64Map& udpRemainingBytesPerSenderThenReceiver,
                                   U32ToU32ToU64Map& udpDatarateSumPerSenderThenReceiver,
                                   U32ToU32ToU32Map& tcpRemainingFlowsPerSenderThenReceiver,
                                   U32ToU32ToU64Map& tcpRemainingBytesPerSenderThenReceiver,
                                   U32ToU32ToU64Map& tcpCWndSumPerSenderThenReceiver)
{
    NS_LOG_FUNCTION(this);

    // initialize maps with empty sub-maps per node id
    for (uint32_t nodeId : m_monitoredNodeIds)
    {
        tcpSenderSrttSumPerSenderThenReceiver[nodeId] = U32ToDMap();
        tcpSenderSrttCountPerSenderThenReceiver[nodeId] = U32ToU32Map();

        udpRemainingFlowsPerSenderThenReceiver[nodeId] = U32ToU32Map();
        udpRemainingBytesPerSenderThenReceiver[nodeId] = U32ToU64Map();
        udpDatarateSumPerSenderThenReceiver[nodeId] = U32ToU64Map();

        tcpRemainingFlowsPerSenderThenReceiver[nodeId] = U32ToU32Map();
        tcpRemainingBytesPerSenderThenReceiver[nodeId] = U32ToU64Map();
        tcpCWndSumPerSenderThenReceiver[nodeId] = U32ToU64Map();
    }

    // Get values from all applications on all nodes
    for (uint32_t nodeId = 0; nodeId < m_simNodes.GetN(); ++nodeId)
    {
        Ptr<Node> node = m_simNodes.Get(nodeId);
        for (uint32_t appIndex = 0; appIndex < node->GetNApplications(); ++appIndex)
        {
            // TCP sender apps
            auto tcpSenderApp = DynamicCast<DisposableBulkSendApplication>(node->GetApplication(appIndex));
            if (tcpSenderApp != nullptr)
            {
                GetTcpSenderAppInfo(nodeId, appIndex, tcpSenderApp,
                                    tcpSenderSrttSumPerSenderThenReceiver, tcpSenderSrttCountPerSenderThenReceiver,
                                    tcpRemainingFlowsPerSenderThenReceiver, tcpRemainingBytesPerSenderThenReceiver,
                                    tcpCWndSumPerSenderThenReceiver);
                continue;
            }

            // UDP sender apps
            auto udpSenderApp = DynamicCast<DisposableOnOffApplication>(node->GetApplication(appIndex));
            if (udpSenderApp != nullptr)
            {
                GetUdpSenderAppInfo(nodeId, appIndex, udpSenderApp,
                                    udpRemainingFlowsPerSenderThenReceiver, udpRemainingBytesPerSenderThenReceiver,
                                    udpDatarateSumPerSenderThenReceiver);
                continue;
            }

            // Other apps are not relevant at the moment
        }
    }
}

void
PackerlMonitor::MonitorNetwork(double elapsedTime, uint64_t monitoringTimeNanoseconds)
{
    NS_LOG_FUNCTION(this);

    // Obtain delay and jitter values from FlowMonitor. Maps are keyed by receiver-then-sender node IDs
    // to facilitate node-centric distribution later on.
    U32ToU32ToDMap maxDelayPerReceiverThenSender;
    U32ToU32ToDMap delaySumPerReceiverThenSender;
    U32ToU32ToU32Map delayCountPerReceiverThenSender;

    U32ToU32ToDMap jitterSumPerReceiverThenSender;
    U32ToU32ToU32Map jitterCountPerReceiverThenSender;
    GetDelayAndJitter(maxDelayPerReceiverThenSender, delaySumPerReceiverThenSender, delayCountPerReceiverThenSender,
                      jitterSumPerReceiverThenSender, jitterCountPerReceiverThenSender);

    // Obtain application information. Maps are keyed by sender-then-receiver node IDs
    // to facilitate node-centric distribution later on.
    U32ToU32ToDMap tcpSenderSrttSumPerSenderThenReceiver;
    U32ToU32ToU32Map tcpSenderSrttCountPerSenderThenReceiver;

    U32ToU32ToU32Map udpRemainingFlowsPerSenderThenReceiver;
    U32ToU32ToU64Map udpRemainingBytesPerSenderThenReceiver;
    U32ToU32ToU64Map udpDatarateSumPerSenderThenReceiver;

    U32ToU32ToU32Map tcpRemainingFlowsPerSenderThenReceiver;
    U32ToU32ToU64Map tcpRemainingBytesPerSenderThenReceiver;
    U32ToU32ToU64Map tcpCWndSumPerSenderThenReceiver;
    GetApplicationInfo(tcpSenderSrttSumPerSenderThenReceiver, tcpSenderSrttCountPerSenderThenReceiver,
                       udpRemainingFlowsPerSenderThenReceiver, udpRemainingBytesPerSenderThenReceiver,
                       udpDatarateSumPerSenderThenReceiver,
                       tcpRemainingFlowsPerSenderThenReceiver, tcpRemainingBytesPerSenderThenReceiver,
                       tcpCWndSumPerSenderThenReceiver);

    // Obtain edge shapshots (they are going to be put into the node snapshots)
    unordered_map<uint32_t, OutgoingEdgeSnapshotMap> outgoingEdgeSnapshotsPerSrcThenDst;
    unordered_map<uint32_t, IncomingEdgeSnapshotMap> incomingEdgeSnapshotsPerDstThenSrc;
    for (auto [src, dst] : m_monitoredEdgeIds)
    {
        Ptr<MonitoringEdge> edge = m_monitoredEdges.at(make_pair(src, dst));
        // Get and update edge data
        uint32_t channelDataRate = edge->getChannelDataRate();
        uint64_t transmittable = static_cast<uint64_t>(round(elapsedTime * static_cast<double>(channelDataRate) / 8.0));
        auto [srcNodeId, srcDevId] = edge->getSrc();
        Ptr<PointToPointNetDevice> srcDev
            = DynamicCast<PointToPointNetDevice> (m_simNodes.Get(srcNodeId)->GetDevice(srcDevId));
        if (srcDev == nullptr)
        {
            NS_FATAL_ERROR(this << " in getRelativeDeviceQueueLoad() (node ID: )" << srcNodeId
                                << ", deviceID: " << srcDevId << ": device does not have a queue!");
        }
        uint32_t txQueueLoad = srcDev->GetQueue()->GetCurrentSize().GetValue();

        // Create edge snapshots, separated by sending and receiving end since different nodes will get this info
        const OutgoingEdgeSnapshot outSnap =
        {
            channelDataRate,                // channelDataRate
            edge->getChannelDelay(),        // channelDelay

            txQueueLoad,                    // txQueueLoad
            edge->getTxQueueCapacity(),     // txQueueCapacity

            edge->getTransmitted(),         // transmitted
            transmittable,                  // transmittable
            edge->getMacTxDropped(),        // macTxDropped

            edge->isLinkUp()                // isLinkUp
        };
        outgoingEdgeSnapshotsPerSrcThenDst[src][dst] = outSnap;

        const IncomingEdgeSnapshot inSnap =
        {
            edge->getPhyRxDropped(),        // phyRxDropped
            edge->isLinkUp()                // isLinkUp
        };
        incomingEdgeSnapshotsPerDstThenSrc[dst][src] = inSnap;
    }

    // Create per-node snapshots, including the previously obtained data
    for (auto& [nodeId, snapshotsByTime] : m_nodeSnapshotsByTime)
    {
        Ptr<MonitoringNode> monitoredNode = m_monitoredNodes.at(nodeId);
        const NodeSnapshot nodeSnap =
        {
            nodeId,                                                 // nodeId

            monitoredNode->GetTxPerReceiver(),                      // txPerReceiver
            monitoredNode->GetReTxPerReceiver(),                    // reTxPerReceiver
            monitoredNode->GetRxPerSender(),                        // rxPerSender
            monitoredNode->GetDeliveredPerSender(),                 // deliveredPerSender
            monitoredNode->GetTtlDroppedPerPair(),                  // ttlDroppedPerPair
            monitoredNode->GetTcpDiscardedPerSender(),              // tcpDiscardedPerSender

            monitoredNode->GetCumulTxPerReceiver(),                 // cumulTxPerReceiver
            monitoredNode->GetCumulDeliveredPerSender(),            // cumulDeliveredPerSender
            monitoredNode->GetCumulTtlDroppedPerPair(),             // cumulTtlDroppedPerPair
            monitoredNode->GetCumulTcpDiscardedPerSender(),         // cumulTcpDiscardedPerSender

            maxDelayPerReceiverThenSender.at(nodeId),               // maxDelayPerSender
            delaySumPerReceiverThenSender.at(nodeId),               // delaySumPerSender
            delayCountPerReceiverThenSender.at(nodeId),             // delayCountPerSender
            jitterSumPerReceiverThenSender.at(nodeId),              // jitterSumPerSender
            jitterCountPerReceiverThenSender.at(nodeId),            // jitterCountPerSender

            tcpSenderSrttSumPerSenderThenReceiver.at(nodeId),       // tcpSenderSrttSumPerReceiver
            tcpSenderSrttCountPerSenderThenReceiver.at(nodeId),     // tcpSenderSrttCountPerReceiver

            udpRemainingFlowsPerSenderThenReceiver.at(nodeId),    // udpRemainingFlowsPerReceiver
            udpRemainingBytesPerSenderThenReceiver.at(nodeId),      // udpRemainingBytesPerReceiver
            udpDatarateSumPerSenderThenReceiver.at(nodeId),         // udpDatarateSumPerReceiver
            tcpRemainingFlowsPerSenderThenReceiver.at(nodeId),    // tcpRemainingFlowsPerReceiver
            tcpRemainingBytesPerSenderThenReceiver.at(nodeId),      // tcpRemainingBytesPerReceiver
            tcpCWndSumPerSenderThenReceiver.at(nodeId),             // tcpCWndSumPerReceiver

            outgoingEdgeSnapshotsPerSrcThenDst.at(nodeId),          // outgoingEdgeSnapshots
            incomingEdgeSnapshotsPerDstThenSrc.at(nodeId),          // incomingEdgeSnapshots
        };
        snapshotsByTime.insert(make_pair(monitoringTimeNanoseconds, nodeSnap));
    }
}

// ===================== Observation ===================================================================================

uint64_t
PackerlMonitor::FindLastMonitoringTime(double now) const
{
    NS_LOG_FUNCTION(this << now);
    NS_ASSERT_MSG(! m_monitoringTimes.empty(), "FindLastMonitoringTime(): no monitoring times available");

    uint64_t nowNanoseconds = static_cast<uint64_t>(now * 1e9);
    auto itTime = upper_bound(m_monitoringTimes.begin(), m_monitoringTimes.end(), nowNanoseconds);
    NS_ASSERT_MSG(itTime != m_monitoringTimes.begin(),
                  "FindLastMonitoringTime(): no monitoring time found that is less or equal to now=" << now);
    return *(--itTime);
}

void
PackerlMonitor::GetNodeSnapshotsGlobal(unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot>& nodeSnapshotsGlobal,
                                       double observationTime)
{
    NS_LOG_FUNCTION(this << observationTime);

    uint64_t lastMonitoringTime = FindLastMonitoringTime(observationTime);
    for (auto const& [nodeId, snapshotsByTime] : m_nodeSnapshotsByTime)
    {
        auto itSnapshot = snapshotsByTime.find(lastMonitoringTime);
        if (itSnapshot == snapshotsByTime.end())
        {
            // TODO include possibility for non-existing node snapshots
            NS_FATAL_ERROR("GetNodeSnapshotsGlobal(): node " << nodeId
                           << " has no snapshot for monitoring time " << lastMonitoringTime);
        }
        nodeSnapshotsGlobal.insert(make_pair(nodeId, itSnapshot->second));
    }
}

void
PackerlMonitor::GetNodeSnapshotsPerObserver(unordered_map<uint32_t,
                                                          unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot>
                                                          >& nodeSnapshotsPerObserver,
                                            double observationTime)
{
    NS_LOG_FUNCTION(this << observationTime);

    for (uint32_t observerId : m_monitoredNodeIds)
    {
        unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot> nodeSnapshotsThisObserver;
        const U32ToDMap& delaysFromObserver = m_communicationDelaysPerSrcThenDst.at(observerId);

        for (auto const& [otherNodeId, snapshotsByTime] : m_nodeSnapshotsByTime)
        {
            // NOTE: this way of obtaining remote observations is a simplification. In reality, we would need to
            //  broadcast node observations over the same network infrastructure that we are simulating on,
            //  which may perturb the actual observation time.
            double effectiveObservationTime = max(0.0, observationTime - 0.001 * delaysFromObserver.at(otherNodeId));
            uint64_t lastMonitoringTime = FindLastMonitoringTime(effectiveObservationTime);
            auto itSnapshot = snapshotsByTime.find(lastMonitoringTime);
            if (itSnapshot == snapshotsByTime.end())
            {
                // TODO include possibility for non-existing node snapshots
                NS_FATAL_ERROR("GetNodeSnapshotsPerObserver(): node " << otherNodeId
                               << " has no snapshot for monitoring time " << lastMonitoringTime);
            }
            nodeSnapshotsThisObserver.insert(make_pair(otherNodeId, itSnapshot->second));
        }
        nodeSnapshotsPerObserver.insert(make_pair(observerId, nodeSnapshotsThisObserver));
    }
}

void
PackerlMonitor::AssembleObservation(long int observerId,
                                    const unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot>& nodeSnapshots,
                                    vector<ObservedGlobal>& observedGlobals,
                                    vector<ObservedNode>& observedNodes,
                                    vector<ObservedEdge>& observedEdges,
                                    vector<ObservedTrafficEntry>& observedTrafficEntries) const
{
    NS_LOG_FUNCTION(this << observerId);

    // ===== Declare observation entries =====

    // traffic matrix entries (only used for global and central node observation)
    U32PairToU64Map txMatrix;

    // src-dst edge entries shipped as observation
    U32PairToU64Map channelDataRatePerEdge;
    U32PairToU32Map channelDelayPerEdge;

    U32PairToU32Map txQueueLoadPerEdge;
    U32PairToU32Map txQueueCapacityPerEdge;

    U32PairToU64Map transmittedPerEdge;
    U32PairToU64Map transmittablePerEdge;
    U32PairToU64Map macTxDroppedPerEdge;
    U32PairToU64Map phyRxDroppedPerEdge;

    U32PairToBoolMap isLinkUpPerEdge;

    // node entries shipped as observation
    U32ToU64Map txAsSender;
    U32ToU64Map txAsReceiver;
    U32ToU64Map reTxAsSender;
    U32ToU64Map reTxAsReceiver;
    U32ToU64Map rxAsSender;
    U32ToU64Map rxAsReceiver;
    U32ToU64Map deliveredAsSender;
    U32ToU64Map deliveredAsReceiver;
    U32ToU64Map ttlDroppedAsSender;
    U32ToU64Map ttlDroppedAsReceiver;
    U32ToU64Map ttlDroppedAsDropper;
    U32ToU64Map tcpDiscardedAsSender;
    U32ToU64Map tcpDiscardedAsReceiver;  // i.e. as dropper

    U32ToDMap maxDelayAsSender;
    U32ToDMap maxDelayAsReceiver;

    U32ToU32Map udpRemainingFlowsAsSender;
    U32ToU32Map udpRemainingFlowsAsReceiver;
    U32ToU64Map udpRemainingFlowBytesAsSender;
    U32ToU64Map udpRemainingFlowBytesAsReceiver;
    U32ToU64Map udpRemainingFlowDatarateSumAsSender;
    U32ToU64Map udpRemainingFlowDatarateSumAsReceiver;
    U32ToU32Map tcpRemainingFlowsAsSender;
    U32ToU32Map tcpRemainingFlowsAsReceiver;
    U32ToU64Map tcpRemainingFlowBytesAsSender;
    U32ToU64Map tcpRemainingFlowBytesAsReceiver;
    U32ToU64Map tcpRemainingFlowCWndSumAsSender;
    U32ToU64Map tcpRemainingFlowCWndSumAsReceiver;

    U32ToU64Map transmittedAsEdgeSrc;
    U32ToU64Map transmittedAsEdgeDst;
    U32ToU64Map transmittableAsEdgeSrc;
    U32ToU64Map transmittableAsEdgeDst;
    U32ToU64Map macTxDroppedAsEdgeSrc;
    U32ToU64Map macTxDroppedAsEdgeDst;
    U32ToU64Map phyRxDroppedAsEdgeSrc;
    U32ToU64Map phyRxDroppedAsEdgeDst;

    U32ToDMap maxLinkUtilizationAsEdgeSrc;
    U32ToDMap maxLinkUtilizationAsEdgeDst;
    U32ToDMap maxTxQueueRelativeLoadAsEdgeSrc;
    U32ToDMap maxTxQueueRelativeLoadAsEdgeDst;

    // node entries only used for further calculations
    U32ToU64Map cumulTxAsSender;
    U32ToU64Map cumulTxAsReceiver;
    U32ToU64Map cumulDeliveredAsSender;
    U32ToU64Map cumulDeliveredAsReceiver;
    U32ToU64Map cumulTtlDroppedAsSender;
    U32ToU64Map cumulTtlDroppedAsReceiver;
    U32ToU64Map cumulTtlDroppedAsDropper;
    U32ToU64Map cumulTcpDiscardedAsSender;
    U32ToU64Map cumulTcpDiscardedAsReceiver;  // i.e. as dropper

    U32ToDMap delaySumAsSender;  // needed to calculate avgDelay later
    U32ToDMap delaySumAsReceiver;  // needed to calculate avgDelay later
    U32ToU32Map delayCountAsSender;  // needed to calculate avgDelay later
    U32ToU32Map delayCountAsReceiver;  // needed to calculate avgDelay later
    U32ToDMap jitterSumAsSender;  // needed to calculate avgJitter later
    U32ToDMap jitterSumAsReceiver;  // needed to calculate avgJitter later
    U32ToU32Map jitterCountAsSender;  // needed to calculate avgJitter later
    U32ToU32Map jitterCountAsReceiver;  // needed to calculate avgJitter later
    U32ToDMap tcpSenderSrttSumAsSender;  // needed to calculate avgTcpSenderSrtt later
    U32ToDMap tcpSenderSrttSumAsReceiver;  // needed to calculate avgTcpSenderSrtt later
    U32ToU32Map tcpSenderSrttCountAsSender;  // needed to calculate avgTcpSenderSrtt later
    U32ToU32Map tcpSenderSrttCountAsReceiver;  // needed to calculate avgTcpSenderSrtt later

    U32ToU64Map txQueueLoadAsEdgeSrc;  // if <= 2**53, we can convert this to double safely
    U32ToU64Map txQueueLoadAsEdgeDst;  // if <= 2**53, we can convert this to double safely
    U32ToU64Map txQueueCapacityAsEdgeSrc;  // if <= 2**53, we can convert this to double safely
    U32ToU64Map txQueueCapacityAsEdgeDst;  // if <= 2**53, we can convert this to double safely

    // global entries (shipped as observation) that will be summed up as we iterate over all node snapshots
    uint64_t tx = 0;
    uint64_t reTx = 0;
    uint64_t rx = 0;
    uint64_t delivered = 0;
    uint64_t ttlDropped = 0;
    uint64_t tcpDiscarded = 0;

    double maxDelay = 0.0;
    uint32_t delayCount = 0;  // needed to calculate avgDelay later
    uint32_t jitterCount = 0;  // needed to calculate avgJitter later
    uint32_t tcpSenderSrttCount = 0;  // needed to calculate avgTcpSenderSrtt later

    uint32_t udpRemainingFlows = 0;
    uint64_t udpRemainingFlowBytes = 0 ;
    uint64_t udpRemainingFlowDatarateSum = 0;
    uint32_t tcpRemainingFlows = 0;
    uint64_t tcpRemainingFlowBytes = 0;
    uint64_t tcpRemainingFlowCWndSum = 0;

    uint64_t txQueueCapacity = 0;  // if <= 2**53, we can convert this to double safely
    uint64_t txQueueLoad = 0;  // if <= 2**53, we can convert this to double safely

    uint64_t transmitted = 0;
    uint64_t transmittable = 0;
    uint64_t macTxDropped = 0;
    uint64_t phyRxDropped = 0;

    double maxLinkUtilization = 0.0;
    double maxTxQueueRelativeLoad = 0.0;

    // global entries (only used for further calculations) that will be summed up as we iterate over all node snapshots
    uint64_t cumulTx = 0;
    uint64_t cumulDelivered = 0;
    uint64_t cumulTtlDropped = 0;
    uint64_t cumulTcpDiscarded = 0;

    double delaySum = 0.0;  // needed to calculate avgDelay later
    double jitterSum = 0.0;  // needed to calculate avgJitter later
    double tcpSenderSrttSum = 0.0;  // needed to calculate avgTcpSenderSrtt later

    // ===== Iterate over all node snapshots (not necessarily in id order) =====
    for (auto const& [fst, nodeSnap] : nodeSnapshots)
    {
        for (auto [snd, val] : nodeSnap.txPerReceiver)
        {
            txAsSender[fst] += val;
            txAsReceiver[snd] += val;
            tx += val;
            if (observerId == m_birdseyeObserverId || static_cast<uint32_t> (observerId) == m_centralNode)
            {
                txMatrix[make_pair(fst, snd)] += val;
            }
        }
        for (auto [snd, val] : nodeSnap.reTxPerReceiver)
        {
            reTxAsSender[fst] += val;
            reTxAsReceiver[snd] += val;
            reTx += val;
        }
        for (auto [snd, val] : nodeSnap.rxPerSender)
        {
            rxAsSender[snd] += val;
            rxAsReceiver[fst] += val;
            rx += val;
        }
        for (auto [snd, val] : nodeSnap.deliveredPerSender)
        {
            deliveredAsSender[snd] += val;
            deliveredAsReceiver[fst] += val;
            delivered += val;
        }
        for (auto [pair, val] : nodeSnap.ttlDroppedPerPair)
        {
            ttlDroppedAsDropper[fst] += val;
            ttlDroppedAsSender[pair.first] += val;
            ttlDroppedAsReceiver[pair.second] += val;
            ttlDropped += val;
        }
        for (auto [snd, val] : nodeSnap.tcpDiscardedPerSender)
        {
            tcpDiscardedAsSender[snd] += val;
            tcpDiscardedAsReceiver[fst] += val;
            tcpDiscarded += val;
        }

        for (auto [snd, val] : nodeSnap.cumulTxPerReceiver)
        {
            cumulTxAsSender[fst] += val;
            cumulTxAsReceiver[snd] += val;
            cumulTx += val;
        }
        for (auto [snd, val] : nodeSnap.cumulDeliveredPerSender)
        {
            cumulDeliveredAsSender[snd] += val;
            cumulDeliveredAsReceiver[fst] += val;
            cumulDelivered += val;
        }
        for (auto [pair, val] : nodeSnap.cumulTtlDroppedPerPair)
        {
            cumulTtlDroppedAsDropper[fst] += val;
            cumulTtlDroppedAsSender[pair.first] += val;
            cumulTtlDroppedAsReceiver[pair.second] += val;
            cumulTtlDropped += val;
        }
        for (auto [snd, val] : nodeSnap.cumulTcpDiscardedPerSender)
        {
            cumulTcpDiscardedAsSender[snd] += val;
            cumulTcpDiscardedAsReceiver[fst] += val;
            cumulTcpDiscarded += val;
        }

        for (auto [snd, val] : nodeSnap.maxDelayPerSender)
        {
            auto it_ps = maxDelayAsSender.find(fst);
            if (it_ps == maxDelayAsSender.end() || val > it_ps->second)
            {
                maxDelayAsSender[fst] = val;
            }
            auto it_pr = maxDelayAsReceiver.find(snd);
            if (it_pr == maxDelayAsReceiver.end() || val > it_pr->second)
            {
                maxDelayAsReceiver[snd] = val;
            }
            maxDelay = max(maxDelay, val);
        }
        for (auto [snd, val] : nodeSnap.delaySumPerSender)
        {
            delaySumAsSender[snd] += val;
            delaySumAsReceiver[fst] += val;
            delaySum += val;
        }
        for (auto [snd, val] : nodeSnap.delayCountPerSender)
        {
            delayCountAsSender[snd] += val;
            delayCountAsReceiver[fst] += val;
            delayCount += val;
        }
        for (auto [snd, val] : nodeSnap.jitterSumPerSender)
        {
            jitterSumAsSender[snd] += val;
            jitterSumAsReceiver[fst] += val;
            jitterSum += val;
        }
        for (auto [snd, val] : nodeSnap.jitterCountPerSender)
        {
            jitterCountAsSender[snd] += val;
            jitterCountAsReceiver[fst] += val;
            jitterCount += val;
        }

        for (auto [snd, val] : nodeSnap.tcpSenderSrttSumPerReceiver)
        {
            tcpSenderSrttSumAsSender[fst] += val;
            tcpSenderSrttSumAsReceiver[snd] += val;
            tcpSenderSrttSum += val;
        }
        for (auto [snd, val] : nodeSnap.tcpSenderSrttCountPerReceiver)
        {
            tcpSenderSrttCountAsSender[fst] += val;
            tcpSenderSrttCountAsReceiver[snd] += val;
            tcpSenderSrttCount += val;
        }

        for (auto [snd, val] : nodeSnap.udpRemainingFlowsPerReceiver)
        {
            udpRemainingFlowsAsSender[fst] += val;
            udpRemainingFlowsAsReceiver[snd] += val;
            udpRemainingFlows += val;
        }
        for (auto [snd, val] : nodeSnap.udpRemainingBytesPerReceiver)
        {
            udpRemainingFlowBytesAsSender[fst] += val;
            udpRemainingFlowBytesAsReceiver[snd] += val;
            udpRemainingFlowBytes += val;
        }
        for (auto [snd, val] : nodeSnap.udpDatarateSumPerReceiver)
        {
            udpRemainingFlowDatarateSumAsSender[fst] += val;
            udpRemainingFlowDatarateSumAsReceiver[snd] += val;
            udpRemainingFlowDatarateSum += val;
        }
        for (auto [snd, val] : nodeSnap.tcpRemainingFlowsPerReceiver)
        {
            tcpRemainingFlowsAsSender[fst] += val;
            tcpRemainingFlowsAsReceiver[snd] += val;
            tcpRemainingFlows += val;
        }
        for (auto [snd, val] : nodeSnap.tcpRemainingBytesPerReceiver)
        {
            tcpRemainingFlowBytesAsSender[fst] += val;
            tcpRemainingFlowBytesAsReceiver[snd] += val;
            tcpRemainingFlowBytes += val;
        }
        for (auto [snd, val] : nodeSnap.tcpCWndSumPerReceiver)
        {
            tcpRemainingFlowCWndSumAsSender[fst] += val;
            tcpRemainingFlowCWndSumAsReceiver[snd] += val;
            tcpRemainingFlowCWndSum += val;
        }

        // loop over outgoing edge snapshots: fst (current snapshot node) is sender, snd (other node) is receiver
        for (auto const& [snd, outgoingEdgeSnap] : nodeSnap.outgoingEdgeSnapshots)
        {
            const U32Pair fstAsEdgeSrc = make_pair(fst, snd);

            // put into matrix entries
            channelDataRatePerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.channelDataRate;
            channelDelayPerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.channelDelay;

            txQueueLoadPerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.txQueueLoad;
            txQueueCapacityPerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.txQueueCapacity;

            transmittedPerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.transmitted;
            transmittablePerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.transmittable;
            macTxDroppedPerEdge[fstAsEdgeSrc] = outgoingEdgeSnap.macTxDropped;

            // link is up if both sender and receiver ends are up
            // (default to true to preserve value, if other snap is not handled yet)
            isLinkUpPerEdge[fstAsEdgeSrc] = getValueOrDefault(isLinkUpPerEdge, fstAsEdgeSrc, true)
                                            && outgoingEdgeSnap.isLinkUp;

            // aggregate into node entries
            txQueueLoadAsEdgeSrc[fst] += outgoingEdgeSnap.txQueueLoad;
            txQueueLoadAsEdgeDst[snd] += outgoingEdgeSnap.txQueueLoad;
            txQueueCapacityAsEdgeSrc[fst] += outgoingEdgeSnap.txQueueCapacity;
            txQueueCapacityAsEdgeDst[snd] += outgoingEdgeSnap.txQueueCapacity;

            transmittedAsEdgeSrc[fst] += outgoingEdgeSnap.transmitted;
            transmittedAsEdgeDst[snd] += outgoingEdgeSnap.transmitted;
            transmittableAsEdgeSrc[fst] += outgoingEdgeSnap.transmittable;
            transmittableAsEdgeDst[snd] += outgoingEdgeSnap.transmittable;
            macTxDroppedAsEdgeSrc[fst] += outgoingEdgeSnap.macTxDropped;
            macTxDroppedAsEdgeDst[snd] += outgoingEdgeSnap.macTxDropped;

            // aggregate into global entries
            txQueueLoad += outgoingEdgeSnap.txQueueLoad;
            txQueueCapacity += outgoingEdgeSnap.txQueueCapacity;

            transmitted += outgoingEdgeSnap.transmitted;
            transmittable += outgoingEdgeSnap.transmittable;
            macTxDropped += outgoingEdgeSnap.macTxDropped;
        }

        // loop over incoming edge snapshots: fst (current snapshot node) is receiver, snd (other node) is sender
        for (auto const& [snd, incomingEdgeSnap] : nodeSnap.incomingEdgeSnapshots)
        {
            const U32Pair sndAsEdgeSrc = make_pair(snd, fst);

            // put into matrix entries
            phyRxDroppedPerEdge[sndAsEdgeSrc] = incomingEdgeSnap.phyRxDropped;

            // link is up if both sender and receiver ends are up
            // (default to true to preserve value, if other snap is not handled yet)
            isLinkUpPerEdge[sndAsEdgeSrc] = getValueOrDefault(isLinkUpPerEdge, sndAsEdgeSrc, true)
                                            && incomingEdgeSnap.isLinkUp;

            // aggregate into node entries
            phyRxDroppedAsEdgeSrc[snd] += incomingEdgeSnap.phyRxDropped;  // phyRxDropped is from snd's perspective
            phyRxDroppedAsEdgeDst[fst] += incomingEdgeSnap.phyRxDropped;  // phyRxDropped is from snd's perspective

            // aggregate into global entries
            phyRxDropped += incomingEdgeSnap.phyRxDropped;
        }
    }

    // ===== Calculate derived metrics and place the observed things into the respective container =====
    //  1. givenUp = ttlDropped + tcpDiscarded;  // (any packets dropped or discarded by the network in this step)
    //  2. cumulGivenUp = cumulTtlDropped + cumulTcpDiscarded;  // (any packets dropped or discarded by the network)
    //  3. inFlight = cumulTx - cumulDelivered - cumulGivenUp;  // (any packets that are still in flight)
    //  4. avgDelay = delayCount > 0 ? delaySum / delayCount : 0.0;
    //  5. avgJitter = jitterCount > 0 ? jitterSum / jitterCount : 0.0;
    //  6. avgTcpSenderSrtt = tcpSenderSrttCount > 0 ? tcpSenderSrttSum / tcpSenderSrttCount : 0.0;
    //  7. linkUtilization = transmittable > 0 ? static_cast<double>(transmitted) / transmittable : 0.0; [cap to 1.0 if m_capLinkUtilization]
    //  8. txQueueRelativeLoad = static_cast<double>(txQueueLoad) / static_cast<double>(txQueueCapacity);
    //  9. givenUpOrDelivered = givenUp + delivered;
    //  10. dropRatio = givenUpOrDelivered > 0 ? static_cast<double>(givenUp) / givenUpOrDelivered : 0.0;
    //  11. inFlightRatio = cumulTx > 0 ? static_cast<double>(inFlight) / cumulTx : 0.0;

    //  [inFlight NOTE] by definition, inFlight values are non-negative, but this applies only to global observation!
    //   for local observers, it can in principle become negative when delivered bytes are registered at the receiver
    //   BEFORE it the receiver sees the corresponding sender tx byte count. We can't prevent this from happening
    //   since the receiver doesn't know WHEN the received packets were sent, so the simplest solution is to just set
    //   inFlight to zero in such cases.
    // ================================================================================================

    // ===== matrix =====
    if (observerId == m_birdseyeObserverId || static_cast<uint32_t> (observerId) == m_centralNode)
    {
        for (uint32_t src : m_monitoredNodeIds)
        {
            for (uint32_t dst : m_monitoredNodeIds)
            {
                observedTrafficEntries.emplace_back(getValueOrDefault(txMatrix, make_pair(src, dst), 0));
            }
        }
    }

    // ===== edge =====
    for (auto [src, dst] : m_monitoredEdgeIds)
    {
        auto edgeId = make_pair(src, dst);

        // --- edge: calculate derived metrics ---
        double linkUtilization = transmittablePerEdge.at(edgeId) > 0
            ? static_cast<double>(transmittedPerEdge.at(edgeId)) / transmittablePerEdge.at(edgeId)
            : 0.0;
        if (m_capLinkUtilization && linkUtilization > 1.0)
        {
            linkUtilization = 1.0;
        }
        auto it_lues = maxLinkUtilizationAsEdgeSrc.find(src);
        if (it_lues == maxLinkUtilizationAsEdgeSrc.end() || linkUtilization > it_lues->second)
        {
            maxLinkUtilizationAsEdgeSrc[src] = linkUtilization;
        }
        auto it_lued = maxLinkUtilizationAsEdgeDst.find(dst);
        if (it_lued == maxLinkUtilizationAsEdgeDst.end() || linkUtilization > it_lued->second)
        {
            maxLinkUtilizationAsEdgeDst[dst] = linkUtilization;
        }
        maxLinkUtilization = max(maxLinkUtilization, linkUtilization);

        double txQueueRelativeLoad = txQueueCapacityPerEdge.at(edgeId) > 0
            ? static_cast<double>(txQueueLoadPerEdge.at(edgeId)) / txQueueCapacityPerEdge.at(edgeId)
            : 0.0;
        auto it_tqles = maxTxQueueRelativeLoadAsEdgeSrc.find(src);
        if (it_tqles == maxTxQueueRelativeLoadAsEdgeSrc.end() || txQueueRelativeLoad > it_tqles->second)
        {
            maxTxQueueRelativeLoadAsEdgeSrc[src] = txQueueRelativeLoad;
        }
        auto it_tqled = maxTxQueueRelativeLoadAsEdgeDst.find(dst);
        if (it_tqled == maxTxQueueRelativeLoadAsEdgeDst.end() || txQueueRelativeLoad > it_tqled->second)
        {
            maxTxQueueRelativeLoadAsEdgeDst[dst] = txQueueRelativeLoad;
        }
        maxTxQueueRelativeLoad = max(maxTxQueueRelativeLoad, txQueueRelativeLoad);

        // --- edge: emplace ---
        observedEdges.emplace_back(
            observerId,                               // observerId
            src,                                      // src
            dst,                                      // dst
            isLinkUpPerEdge.at(edgeId),               // isLinkUp

            channelDataRatePerEdge.at(edgeId),        // channelDataRate
            channelDelayPerEdge.at(edgeId),           // channelDelay
            txQueueCapacityPerEdge.at(edgeId),        // txQueueCapacity
            txQueueLoadPerEdge.at(edgeId),            // txQueueLoad

            transmittedPerEdge.at(edgeId),            // transmitted
            transmittablePerEdge.at(edgeId),          // transmittable
            macTxDroppedPerEdge.at(edgeId),           // macTxDropped
            phyRxDroppedPerEdge.at(edgeId),           // phyRxDropped

            linkUtilization,                          // linkUtilization
            txQueueRelativeLoad                       // txQueueRelativeLoad
        );
    }

    // ===== node =====
    for (uint32_t nodeId : m_monitoredNodeIds)
    {
        // --- node: calculate derived metrics ---
        uint64_t givenUpAsSender = getValueOrDefault(ttlDroppedAsSender, nodeId, 0)
                                   + getValueOrDefault(tcpDiscardedAsSender, nodeId, 0);
        uint64_t givenUpAsReceiver = getValueOrDefault(ttlDroppedAsReceiver, nodeId, 0)
                                     + getValueOrDefault(tcpDiscardedAsReceiver, nodeId, 0);
        uint64_t givenUpAsDropper = getValueOrDefault(ttlDroppedAsDropper, nodeId, 0)
                                    + getValueOrDefault(tcpDiscardedAsReceiver, nodeId, 0);  // tcpDiscarded: receiver = dropper

        uint64_t cumulGivenUpAsSender = getValueOrDefault(cumulTtlDroppedAsSender, nodeId, 0) + getValueOrDefault(cumulTcpDiscardedAsSender, nodeId, 0);
        uint64_t curCumulTxAsSender = getValueOrDefault(cumulTxAsSender, nodeId, 0);
        uint64_t curCumulDeliveredAsSender = getValueOrDefault(cumulDeliveredAsSender, nodeId, 0);

        uint64_t inFlightAsSender;
        if (curCumulTxAsSender < curCumulDeliveredAsSender + cumulGivenUpAsSender)
        {
            // for global observer, inFlight must be non-negative
            if (observerId == m_birdseyeObserverId)
            {
                NS_FATAL_ERROR("AssembleObservation() observerId " << observerId
                               << ": inFlightAsSender would underflow for node " << nodeId
                               << ". CumulTx: " << curCumulTxAsSender
                               << ", CumulDelivered: " << curCumulDeliveredAsSender
                               << ", CumulGivenUp: " << cumulGivenUpAsSender);
            }
            // for local observer, set inFlight to zero (see [inFlight NOTE] above)
            inFlightAsSender = 0;
        }
        else
        {
            inFlightAsSender = curCumulTxAsSender - curCumulDeliveredAsSender - cumulGivenUpAsSender;
        }

        uint64_t cumulGivenUpAsReceiver = getValueOrDefault(cumulTtlDroppedAsReceiver, nodeId, 0)
                                          + getValueOrDefault(cumulTcpDiscardedAsReceiver, nodeId, 0);
        uint64_t curCumulTxAsReceiver = getValueOrDefault(cumulTxAsReceiver, nodeId, 0);
        uint64_t curCumulDeliveredAsReceiver = getValueOrDefault(cumulDeliveredAsReceiver, nodeId, 0);

        uint64_t inFlightAsReceiver;
        if (curCumulTxAsReceiver < curCumulDeliveredAsReceiver + cumulGivenUpAsReceiver)
        {
            // for global observer, inFlight must be non-negative
            if (observerId == m_birdseyeObserverId)
            {
                NS_FATAL_ERROR("AssembleObservation() observerId " << observerId
                               << ": inFlightAsReceiver would underflow for node " << nodeId
                               << ". CumulTx: " << curCumulTxAsReceiver
                               << ", CumulDelivered: " << curCumulDeliveredAsReceiver
                               << ", CumulGivenUp: " << cumulGivenUpAsReceiver);
            }
            // for local observer, set inFlight to zero (see [inFlight NOTE] above)
            inFlightAsReceiver = 0;
        }
        else
        {
            inFlightAsReceiver = curCumulTxAsReceiver - curCumulDeliveredAsReceiver - cumulGivenUpAsReceiver;
        }

        uint32_t delayCountAsSndr = getValueOrDefault(delayCountAsSender, nodeId, 0);
        double avgDelayAsSender = delayCountAsSndr > 0
            ? getValueOrDefault(delaySumAsSender, nodeId, 0.0) / delayCountAsSndr
            : 0.0;
        uint32_t delayCountAsRcvr = getValueOrDefault(delayCountAsReceiver, nodeId, 0);
        double avgDelayAsReceiver = delayCountAsRcvr > 0
            ? getValueOrDefault(delaySumAsReceiver, nodeId, 0.0) / delayCountAsRcvr
            : 0.0;
        uint32_t jitterCountAsSndr = getValueOrDefault(jitterCountAsSender, nodeId, 0);
        double avgJitterAsSender = jitterCountAsSndr > 0
            ? getValueOrDefault(jitterSumAsSender, nodeId, 0.0) / jitterCountAsSndr
            : 0.0;
        uint32_t jitterCountAsRcvr = getValueOrDefault(jitterCountAsReceiver, nodeId, 0);
        double avgJitterAsReceiver = jitterCountAsRcvr > 0
            ? getValueOrDefault(jitterSumAsReceiver, nodeId, 0.0) / jitterCountAsRcvr
            : 0.0;
        uint32_t tcpSenderSrttCountAsSndr = getValueOrDefault(tcpSenderSrttCountAsSender, nodeId, 0);
        double avgTcpSenderSrttAsSender = tcpSenderSrttCountAsSndr > 0
            ? getValueOrDefault(tcpSenderSrttSumAsSender, nodeId, 0.0) / tcpSenderSrttCountAsSndr
            : 0.0;
        uint32_t tcpSenderSrttCountAsRcvr = getValueOrDefault(tcpSenderSrttCountAsReceiver, nodeId, 0);
        double avgTcpSenderSrttAsReceiver = tcpSenderSrttCountAsRcvr > 0
            ? getValueOrDefault(tcpSenderSrttSumAsReceiver, nodeId, 0.0) / tcpSenderSrttCountAsRcvr
            : 0.0;

        uint64_t transmittableThisEdgeSrc = getValueOrDefault(transmittableAsEdgeSrc, nodeId, 0);
        double avgLinkUtilizationAsEdgeSrc = transmittableThisEdgeSrc > 0
            ? static_cast<double>(getValueOrDefault(transmittedAsEdgeSrc, nodeId, 0)) / transmittableThisEdgeSrc
            : 0.0;
        uint64_t transmittableThisEdgeDst = getValueOrDefault(transmittableAsEdgeDst, nodeId, 0);
        double avgLinkUtilizationAsEdgeDst = transmittableThisEdgeDst > 0
            ? static_cast<double>(getValueOrDefault(transmittedAsEdgeDst, nodeId, 0)) / transmittableThisEdgeDst
            : 0.0;
        uint64_t txQueueCapacityThisEdgeSrc = getValueOrDefault(txQueueCapacityAsEdgeSrc, nodeId, 0);
        double avgTxQueueRelativeLoadAsEdgeSrc = txQueueCapacityThisEdgeSrc > 0
            ? static_cast<double>(getValueOrDefault(txQueueLoadAsEdgeSrc, nodeId, 0)) / txQueueCapacityThisEdgeSrc
            : 0.0;
        uint64_t txQueueCapacityThisEdgeDst = getValueOrDefault(txQueueCapacityAsEdgeDst, nodeId, 0);
        double avgTxQueueRelativeLoadAsEdgeDst = txQueueCapacityThisEdgeDst > 0
            ? static_cast<double>(getValueOrDefault(txQueueLoadAsEdgeDst, nodeId, 0)) / txQueueCapacityThisEdgeDst
            : 0.0;

        uint64_t droppedOrDeliveredAsSender = givenUpAsSender + getValueOrDefault(deliveredAsSender, nodeId, 0);
        double dropRatioAsSender = droppedOrDeliveredAsSender > 0
            ? static_cast<double>(givenUpAsSender) / droppedOrDeliveredAsSender
            : 0.0;
        uint64_t cumulTxThisSender = getValueOrDefault(cumulTxAsSender, nodeId, 0);
        double inFlightRatioAsSender = cumulTxThisSender > 0
            ? static_cast<double>(inFlightAsSender) / cumulTxThisSender
            : 0.0;

        uint64_t droppedOrDeliveredAsReceiver = givenUpAsReceiver + getValueOrDefault(deliveredAsReceiver, nodeId, 0);
        double dropRatioAsReceiver = droppedOrDeliveredAsReceiver > 0
            ? static_cast<double>(givenUpAsReceiver) / droppedOrDeliveredAsReceiver
            : 0.0;
        uint64_t cumulTxThisReceiver = getValueOrDefault(cumulTxAsReceiver, nodeId, 0);
        double inFlightRatioAsReceiver = cumulTxThisReceiver > 0
            ? static_cast<double>(inFlightAsReceiver) / cumulTxThisReceiver
            : 0.0;

        // --- node: emplace ---
        observedNodes.emplace_back(
            observerId,                                                             // observerId
            nodeId,                                                                 // nodeId

            getValueOrDefault(txAsSender, nodeId, 0),                               // txAsSender
            getValueOrDefault(txAsReceiver, nodeId, 0),                             // txAsReceiver
            getValueOrDefault(reTxAsSender, nodeId, 0),                             // reTxAsSender
            getValueOrDefault(reTxAsReceiver, nodeId, 0),                           // reTxAsReceiver
            inFlightAsSender,                                                       // inFlightAsSender
            inFlightAsReceiver,                                                     // inFlightAsReceiver
            getValueOrDefault(rxAsSender, nodeId, 0),                               // rxAsSender
            getValueOrDefault(rxAsReceiver, nodeId, 0),                             // rxAsReceiver
            getValueOrDefault(deliveredAsSender, nodeId, 0),                        // deliveredAsSender
            getValueOrDefault(deliveredAsReceiver, nodeId, 0),                      // deliveredAsReceiver
            getValueOrDefault(ttlDroppedAsSender, nodeId, 0),                       // ttlDroppedAsSender
            getValueOrDefault(ttlDroppedAsReceiver, nodeId, 0),                     // ttlDroppedAsReceiver
            getValueOrDefault(ttlDroppedAsDropper, nodeId, 0),                      // ttlDroppedAsDropper
            getValueOrDefault(tcpDiscardedAsSender, nodeId, 0),                     // tcpDiscardedAsSender
            getValueOrDefault(tcpDiscardedAsReceiver, nodeId, 0),                   // tcpDiscardedAsReceiver
            givenUpAsSender,                                                        // givenUpAsSender
            givenUpAsReceiver,                                                      // givenUpAsReceiver
            givenUpAsDropper,                                                       // givenUpAsDropper

            getValueOrDefault(maxDelayAsSender, nodeId, 0.0),                       // maxDelayAsSender
            getValueOrDefault(maxDelayAsReceiver, nodeId, 0.0),                     // maxDelayAsReceiver
            avgDelayAsSender,                                                       // avgDelayAsSender
            avgDelayAsReceiver,                                                     // avgDelayAsReceiver
            avgJitterAsSender,                                                      // avgJitterAsSender
            avgJitterAsReceiver,                                                    // avgJitterAsReceiver
            avgTcpSenderSrttAsSender,                                               // avgTcpSenderSrttAsSender
            avgTcpSenderSrttAsReceiver,                                             // avgTcpSenderSrttAsReceiver

            getValueOrDefault(udpRemainingFlowsAsSender, nodeId, 0),              // udpRemainingFlowsAsSender
            getValueOrDefault(udpRemainingFlowsAsReceiver, nodeId, 0),            // udpRemainingFlowsAsReceiver
            getValueOrDefault(udpRemainingFlowBytesAsSender, nodeId, 0),          // udpRemainingFlowBytesAsSender
            getValueOrDefault(udpRemainingFlowBytesAsReceiver, nodeId, 0),        // udpRemainingFlowBytesAsReceiver
            getValueOrDefault(udpRemainingFlowDatarateSumAsSender, nodeId, 0),    // udpRemainingFlowDatarateSumAsSender
            getValueOrDefault(udpRemainingFlowDatarateSumAsReceiver, nodeId, 0),  // udpRemainingFlowDatarateSumAsReceiver
            getValueOrDefault(tcpRemainingFlowsAsSender, nodeId, 0),              // tcpRemainingFlowsAsSender
            getValueOrDefault(tcpRemainingFlowsAsReceiver, nodeId, 0),            // tcpRemainingFlowsAsReceiver
            getValueOrDefault(tcpRemainingFlowBytesAsSender, nodeId, 0),          // tcpRemainingFlowBytesAsSender
            getValueOrDefault(tcpRemainingFlowBytesAsReceiver, nodeId, 0),        // tcpRemainingFlowBytesAsReceiver
            getValueOrDefault(tcpRemainingFlowCWndSumAsSender, nodeId, 0),        // tcpRemainingFlowCWndSumAsSender
            getValueOrDefault(tcpRemainingFlowCWndSumAsReceiver, nodeId, 0),      // tcpRemainingFlowCWndSumAsReceiver

            getValueOrDefault(transmittedAsEdgeSrc, nodeId, 0),                     // transmittedAsEdgeSrc
            getValueOrDefault(transmittedAsEdgeDst, nodeId, 0),                     // transmittedAsEdgeDst
            getValueOrDefault(transmittableAsEdgeSrc, nodeId, 0),                   // transmittableAsEdgeSrc
            getValueOrDefault(transmittableAsEdgeDst, nodeId, 0),                   // transmittableAsEdgeDst
            getValueOrDefault(macTxDroppedAsEdgeSrc, nodeId, 0),                    // macTxDroppedAsEdgeSrc
            getValueOrDefault(macTxDroppedAsEdgeDst, nodeId, 0),                    // macTxDroppedAsEdgeDst
            getValueOrDefault(phyRxDroppedAsEdgeSrc, nodeId, 0),                    // phyRxDroppedAsEdgeSrc
            getValueOrDefault(phyRxDroppedAsEdgeDst, nodeId, 0),                    // phyRxDroppedAsEdgeDst

            dropRatioAsSender,                                                      // dropRatioAsSender
            dropRatioAsReceiver,                                                    // dropRatioAsReceiver
            inFlightRatioAsSender,                                                  // inFlightRatioAsSender
            inFlightRatioAsReceiver,                                                // inFlightRatioAsReceiver

            getValueOrDefault(maxLinkUtilizationAsEdgeSrc, nodeId, 0.0),            // maxLinkUtilizationAsEdgeSrc
            getValueOrDefault(maxLinkUtilizationAsEdgeDst, nodeId, 0.0),            // maxLinkUtilizationAsEdgeDst
            avgLinkUtilizationAsEdgeSrc,                                            // avgLinkUtilizationAsEdgeSrc
            avgLinkUtilizationAsEdgeDst,                                            // avgLinkUtilizationAsEdgeDst
            getValueOrDefault(maxTxQueueRelativeLoadAsEdgeSrc, nodeId, 0.0),        // maxTxQueueRelativeLoadAsEdgeSrc
            getValueOrDefault(maxTxQueueRelativeLoadAsEdgeDst, nodeId, 0.0),        // maxTxQueueRelativeLoadAsEdgeDst
            avgTxQueueRelativeLoadAsEdgeSrc,                                        // avgTxQueueRelativeLoadAsEdgeSrc
            avgTxQueueRelativeLoadAsEdgeDst                                         // avgTxQueueRelativeLoadAsEdgeDst
        );
    }

    // ===== global: calculate derived metrics ===== //
    uint64_t givenUp = ttlDropped + tcpDiscarded;
    uint64_t cumulGivenUp = cumulTtlDropped + cumulTcpDiscarded;
    uint64_t inFlight;
    if (cumulTx < cumulDelivered + cumulGivenUp)
    {
        // for global observer, inFlight must be non-negative
        if (observerId == m_birdseyeObserverId)
        {
            NS_FATAL_ERROR("AssembleObservation() global: inFlight would underflow. CumulTx: " << cumulTx
                           << ", CumulDelivered: " << cumulDelivered
                           << ", CumulGivenUp: " << cumulGivenUp);
        }
        // for local observer, set inFlight to zero (see [inFlight NOTE] above)
        else
        {
            inFlight = 0;
        }
    }
    else
    {
        inFlight = cumulTx - cumulDelivered - cumulGivenUp;
    }
    double avgDelay = delayCount > 0 ? delaySum / delayCount : 0.0;
    double avgJitter = jitterCount > 0 ? jitterSum / jitterCount : 0.0;
    double avgTcpSenderSrtt = tcpSenderSrttCount > 0 ? tcpSenderSrttSum / tcpSenderSrttCount : 0.0;
    double avgLinkUtilization = transmittable > 0 ? static_cast<double>(transmitted) / transmittable : 0.0;
    if (m_capLinkUtilization && avgLinkUtilization > 1.0)
    {
        avgLinkUtilization = 1.0;
    }
    double avgTxQueueRelativeLoad = txQueueCapacity > 0 ? static_cast<double>(txQueueLoad) / txQueueCapacity : 0.0;
    uint64_t givenUpOrDelivered = givenUp + delivered;
    double dropRatio = givenUpOrDelivered > 0 ? static_cast<double>(givenUp) / givenUpOrDelivered : 0.0;
    double inFlightRatio = cumulTx > 0 ? static_cast<double>(inFlight) / cumulTx : 0.0;

    // ===== global: emplace =====
    observedGlobals.emplace_back(
        observerId,                         // .observerId

        tx,                                 // .tx
        reTx,                               // .reTx
        inFlight,                           // .inFlight
        rx,                                 // .rx
        delivered,                          // .delivered
        ttlDropped,                         // .ttlDropped
        tcpDiscarded,                       // .tcpDiscarded
        givenUp,                            // .givenUp

        maxDelay,                           // .maxDelay
        delayCount,                         // .delayCount
        avgDelay,                           // .avgDelay
        jitterCount,                        // .jitterCount
        avgJitter,                          // .avgJitter
        tcpSenderSrttCount,                 // .tcpSenderSrttCount
        avgTcpSenderSrtt,                   // .avgTcpSenderSrtt

        udpRemainingFlows,                // .udpRemainingFlows
        udpRemainingFlowBytes,            // .udpRemainingFlowBytes
        udpRemainingFlowDatarateSum,      // .udpRemainingFlowDatarateSum
        tcpRemainingFlows,                // .tcpRemainingFlows
        tcpRemainingFlowBytes,            // .tcpRemainingFlowBytes
        tcpRemainingFlowCWndSum,          // .tcpRemainingFlowCWndSum

        txQueueCapacity,                    // .txQueueCapacity
        txQueueLoad,                        // .txQueueLoad

        transmitted,                        // .transmitted
        transmittable,                      // .transmittable
        macTxDropped,                       // .macTxDropped
        phyRxDropped,                       // .phyRxDropped

        dropRatio,                          // .dropRatio
        inFlightRatio,                      // .inFlightRatio
        maxLinkUtilization,                 // .maxLinkUtilization
        avgLinkUtilization,                 // .avgLinkUtilization
        maxTxQueueRelativeLoad,             // .maxTxQueueRelativeLoad
        avgTxQueueRelativeLoad              // .avgTxQueueRelativeLoad
    );
}

void
PackerlMonitor::PrintNodeSnapshot(long int observerId,
                                  const PackerlMonitor::NodeSnapshot& nodeSnap, double monitoringTime) const
{
    NS_LOG_FUNCTION(this);

    // inline function to print maps via NS_LOG_DEBUG, just like PrintMap()
    auto PrintMap = [this](const auto& map) {
        if (map.empty())
        {
            NS_LOG_DEBUG("| - (empty)");
            return;
        }
        for (const auto& entry : map)
        {
            NS_LOG_DEBUG("| - Key: " << entry.first << ", Value: " << entry.second);
        }
    };

    uint32_t monitoringTimeMs = static_cast<uint32_t>(monitoringTime * 1000);
    NS_LOG_DEBUG("+------------- ObserverId: " << observerId << ", NodeSnapshot " << nodeSnap.nodeId
                << " @ t=" << monitoringTimeMs << "ms -------------------------------------");
    NS_LOG_DEBUG("| " << "");
    NS_LOG_DEBUG("| " << "txPerReceiver: ");
    PrintMap(nodeSnap.txPerReceiver);
    NS_LOG_DEBUG("| " << "reTxPerReceiver: ");
    PrintMap(nodeSnap.reTxPerReceiver);
    NS_LOG_DEBUG("| " << "rxPerSender: ");
    PrintMap(nodeSnap.rxPerSender);
    NS_LOG_DEBUG("| " << "deliveredPerSender: ");
    PrintMap(nodeSnap.deliveredPerSender);
    NS_LOG_DEBUG("| " << "ttlDroppedPerPair: ");
    PrintMap(nodeSnap.ttlDroppedPerPair);
    NS_LOG_DEBUG("| " << "tcpDiscardedPerSender: ");
    PrintMap(nodeSnap.tcpDiscardedPerSender);
    NS_LOG_DEBUG("| " << "");
    NS_LOG_DEBUG("| " << "cumulTxPerReceiver: ");
    PrintMap(nodeSnap.cumulTxPerReceiver);
    NS_LOG_DEBUG("| " << "cumulDeliveredPerSender: ");
    PrintMap(nodeSnap.cumulDeliveredPerSender);
    NS_LOG_DEBUG("| " << "cumulTtlDroppedPerPair: ");
    PrintMap(nodeSnap.cumulTtlDroppedPerPair);
    NS_LOG_DEBUG("| " << "cumulTcpDiscardedPerSender: ");
    PrintMap(nodeSnap.cumulTcpDiscardedPerSender);
    NS_LOG_DEBUG("| " << "");
    NS_LOG_DEBUG("| " << "maxDelayPerSender: ");
    PrintMap(nodeSnap.maxDelayPerSender);
    NS_LOG_DEBUG("| " << "delaySumPerSender: ");
    PrintMap(nodeSnap.delaySumPerSender);
    NS_LOG_DEBUG("| " << "delayCountPerSender: ");
    PrintMap(nodeSnap.delayCountPerSender);
    NS_LOG_DEBUG("| " << "jitterSumPerSender: ");
    PrintMap(nodeSnap.jitterSumPerSender);
    NS_LOG_DEBUG("| " << "jitterCountPerSender: ");
    PrintMap(nodeSnap.jitterCountPerSender);
    NS_LOG_DEBUG("| " << "");
    NS_LOG_DEBUG("| " << "tcpSenderSrttSumPerReceiver: ");
    PrintMap(nodeSnap.tcpSenderSrttSumPerReceiver);
    NS_LOG_DEBUG("| " << "tcpSenderSrttCountPerReceiver: ");
    PrintMap(nodeSnap.tcpSenderSrttCountPerReceiver);
    NS_LOG_DEBUG("| " << "");
    NS_LOG_DEBUG("| " << "udpRemainingFlowsPerReceiver: ");
    PrintMap(nodeSnap.udpRemainingFlowsPerReceiver);
    NS_LOG_DEBUG("| " << "udpRemainingBytesPerReceiver: ");
    PrintMap(nodeSnap.udpRemainingBytesPerReceiver);
    NS_LOG_DEBUG("| " << "udpDatarateSumPerReceiver: ");
    PrintMap(nodeSnap.udpDatarateSumPerReceiver);
    NS_LOG_DEBUG("| " << "tcpRemainingFlowsPerReceiver: ");
    PrintMap(nodeSnap.tcpRemainingFlowsPerReceiver);
    NS_LOG_DEBUG("| " << "tcpRemainingBytesPerReceiver: ");
    PrintMap(nodeSnap.tcpRemainingBytesPerReceiver);
    NS_LOG_DEBUG("| " << "tcpCWndSumPerReceiver: ");
    PrintMap(nodeSnap.tcpCWndSumPerReceiver);
    NS_LOG_DEBUG("| " << "");
    for (auto const& [edgeDst, outgoingEdgeSnap] : nodeSnap.outgoingEdgeSnapshots)
    {
        NS_LOG_DEBUG("| " << "Outgoing Edge to " << edgeDst << ":");
        NS_LOG_DEBUG("| -  txQueueLoad: " << outgoingEdgeSnap.txQueueLoad);
        NS_LOG_DEBUG("| -  txQueueCapacity: " << outgoingEdgeSnap.txQueueCapacity);
        NS_LOG_DEBUG("| -  transmitted: " << outgoingEdgeSnap.transmitted);
        NS_LOG_DEBUG("| -  transmittable: " << outgoingEdgeSnap.transmittable);
        NS_LOG_DEBUG("| -  macTxDropped: " << outgoingEdgeSnap.macTxDropped);
        NS_LOG_DEBUG("| " << "");
    }
    for (auto const& [edgeSrc, incomingEdgeSnap] : nodeSnap.incomingEdgeSnapshots)
    {
        NS_LOG_DEBUG("| " << "Incoming Edge from " << edgeSrc << ":");
        NS_LOG_DEBUG("| -  phyRxDropped: " << incomingEdgeSnap.phyRxDropped);
        NS_LOG_DEBUG("| " << "");
    }
    NS_LOG_DEBUG("+------------------------------------------------------------------------------------------");
}

void
PackerlMonitor::PrintObservedGlobal(const ObservedGlobal& global, double monitoringTime) const
{
    NS_LOG_FUNCTION(this);

    long int observerId = global.observerId;
    uint32_t monitoringTimeMs = static_cast<uint32_t>(monitoringTime * 1000);
    NS_LOG_DEBUG("+------------- ObserverId: " << observerId
                << ", Observed Global @ t=" << monitoringTimeMs << "ms  ----------------------------------------");
    NS_LOG_DEBUG("| " << std::setw(35) << "");
    NS_LOG_DEBUG("| " << std::setw(35) << "Sent out into routing: " << std::setw(15) << global.tx << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "Retransmitted: " << std::setw(15) << global.reTx << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "In-Flight (incl. past steps): " << std::setw(15) << global.inFlight << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "Received: " << std::setw(15) << global.rx << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "Delivered (goodput): " << std::setw(15) << global.delivered << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "TTL-Dropped: " << std::setw(15) << global.ttlDropped << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "TCP-Discarded: " << std::setw(15) << global.tcpDiscarded << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "Given up: " << std::setw(15) << global.givenUp << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "");
    NS_LOG_DEBUG("| " << std::setw(35) << "Max Delay: " << std::setw(15) << global.maxDelay * 1000 << " ms");
    NS_LOG_DEBUG("| " << std::setw(35) << "Delay Count: " << std::setw(15) << global.delayCount);
    NS_LOG_DEBUG("| " << std::setw(35) << "Avg Delay: " << std::setw(15) << global.avgDelay * 1000 << " ms");
    NS_LOG_DEBUG("| " << std::setw(35) << "Jitter Count: " << std::setw(15) << global.jitterCount);
    NS_LOG_DEBUG("| " << std::setw(35) << "Avg Jitter: " << std::setw(15) << global.avgJitter * 1000 << " ms");
    NS_LOG_DEBUG("| " << std::setw(35) << "TCP Sender SRTT Count: " << std::setw(15) << global.tcpSenderSrttCount);
    NS_LOG_DEBUG("| " << std::setw(35) << "Avg Sender TCP smoothed RTT: " << std::setw(15) << global.avgTcpSenderSrtt * 1000 << " ms");
    NS_LOG_DEBUG("| " << std::setw(35) << "");
    NS_LOG_DEBUG("| " << std::setw(35) << "UDP Active Flows: " << std::setw(15) << global.udpRemainingFlows << " flows");
    NS_LOG_DEBUG("| " << std::setw(35) << "UDP Remaining Flow Bytes: " << std::setw(15) << global.udpRemainingFlowBytes << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "UDP Rem. Flow Datarate Sum: " << std::setw(15) << global.udpRemainingFlowDatarateSum << " bps");
    NS_LOG_DEBUG("| " << std::setw(35) << "TCP Active Flows: " << std::setw(15) << global.tcpRemainingFlows << " flows");
    NS_LOG_DEBUG("| " << std::setw(35) << "TCP Remaining Flow Bytes: " << std::setw(15) << global.tcpRemainingFlowBytes << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "TCP Rem. Flow CWnd Sum: " << std::setw(15) << global.tcpRemainingFlowCWndSum << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "");
    NS_LOG_DEBUG("| " << std::setw(35) << "Transmitted over links (gross): " << std::setw(15) << global.transmitted << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "Transmittable over links: " << std::setw(15) << global.transmittable << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "MAC-Tx-Dropped: " << std::setw(15) << global.macTxDropped << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "PHY-Rx-Dropped: " << std::setw(15) << global.phyRxDropped << " bytes");
    NS_LOG_DEBUG("| " << std::setw(35) << "");
    NS_LOG_DEBUG("| " << std::setw(35) << "Drop Ratio: " << std::setw(15) << global.dropRatio * 100 << " %");
    NS_LOG_DEBUG("| " << std::setw(35) << "In-Flight Ratio (inFlight / tx): " << std::setw(15) << global.inFlightRatio * 100 << " %");
    NS_LOG_DEBUG("| " << std::setw(35) << "Max Link Utilization: " << std::setw(15) << global.maxLinkUtilization * 100 << " %");
    NS_LOG_DEBUG("| " << std::setw(35) << "Avg Link Utilization: " << std::setw(15) << global.avgLinkUtilization * 100 << " %");
    NS_LOG_DEBUG("| " << std::setw(35) << "Max TxQueue Relative Load: " << std::setw(15) << global.maxTxQueueRelativeLoad * 100 << " %");
    NS_LOG_DEBUG("| " << std::setw(35) << "Avg TxQueue Relative Load: " << std::setw(15) << global.avgTxQueueRelativeLoad * 100 << " %");
    NS_LOG_DEBUG("| " << std::setw(35) << "");
    NS_LOG_DEBUG("+-----------------------------------------------------------------------------------------------");
}

void
PackerlMonitor::ResetMonitor(double monitoringTime)
{
    NS_LOG_FUNCTION(this);

    for (auto& [_, node] : m_monitoredNodes)
    {
        node->resetStats();
    }
    for (auto& [_, edge] : m_monitoredEdges)
    {
        edge->resetStats();
    }
    m_flowMonitor->ResetAllStats();
    m_nodeRewarder->ResetRewards();
}

NetworkObservation
PackerlMonitor::ObserveNetwork()
{
    NS_LOG_FUNCTION(this);

    // Monitor the network
    double monitoringTime = Simulator::Now().GetSeconds();
    double elapsedTime = monitoringTime - (m_monitoringTimes.empty() ? 0 : m_monitoringTimes.back()) * 1e-9;
    uint64_t monitoringTimeNanoseconds = static_cast<uint64_t>(std::floor(monitoringTime * 1e9));
    m_monitoringTimes.push_back(monitoringTimeNanoseconds);
    MonitorNetwork(elapsedTime, monitoringTimeNanoseconds);

    NS_ASSERT_MSG(m_monitoredNodes.find(m_centralNode) != m_monitoredNodes.end(), "Central node not monitored!");

    // Prepare the constructs that will be filled with the observed data
    uint32_t V = m_monitoredNodes.size();
    uint32_t E = m_monitoredEdges.size();

    ObservationInfo observedInfo = {monitoringTime, elapsedTime, V, E};
    vector<ObservedGlobal> observedGlobals;
    vector<ObservedNode> observedNodes;
    vector<ObservedEdge> observedEdges;
    vector<ObservedTrafficEntry> observedTrafficEntries;
    vector<ObservedReward> observedRewards;

    observedGlobals.reserve(V+1);
    observedNodes.reserve((V+1) * V);
    observedEdges.reserve((V+1) * E);
    observedRewards.reserve(V+1);
    observedTrafficEntries.reserve(2 * V*V);  // traffic matrix entries for 'global' and central node only

    // Observe the monitored network state (node-centric observations respect channel delays)
    unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot> nodeSnapshotsGlobal;
    GetNodeSnapshotsGlobal(nodeSnapshotsGlobal, monitoringTime);
    AssembleObservation(-1, nodeSnapshotsGlobal, observedGlobals, observedNodes, observedEdges, observedTrafficEntries);

    unordered_map<uint32_t, unordered_map<uint32_t, const PackerlMonitor::NodeSnapshot>> nodeSnapshotsPerObserver;
    GetNodeSnapshotsPerObserver(nodeSnapshotsPerObserver, monitoringTime);
    for (uint32_t observerId : m_monitoredNodeIds)
    {
        AssembleObservation(observerId, nodeSnapshotsPerObserver.at(observerId),
                            observedGlobals, observedNodes, observedEdges, observedTrafficEntries);
    }

    // debug prints
    if (g_log.IsEnabled(ns3::LOG_DEBUG))
    {
        // globally viewed node snapshots
        for (uint32_t nodeId : m_monitoredNodeIds)
        {
            PrintNodeSnapshot((long int) -1, nodeSnapshotsGlobal.at(nodeId), monitoringTime);
        }
        // node snapshots per local observer
        for (uint32_t observerId : m_monitoredNodeIds)
        {
            for (uint32_t nodeId : m_monitoredNodeIds)
            {
                PrintNodeSnapshot((long int) observerId,
                                  nodeSnapshotsPerObserver.at(observerId).at(nodeId), monitoringTime);
            }
        }
        // global observations per observer (starting with global)
        for (const ObservedGlobal& observedGlobal : observedGlobals)
        {
            PrintObservedGlobal(observedGlobal, monitoringTime);
        }
    }

    // Obtain rewards
    observedRewards.emplace_back(-1, m_globalRewarder->CalculateReward(observedGlobals[0]));
    const U32ToDMap& nodeRewards = m_nodeRewarder->GetNodeRewards();
    for (uint32_t nodeId : m_monitoredNodeIds)
    {
        observedRewards.emplace_back(nodeId, nodeRewards.at(nodeId));
    }

    // finalize
    ResetMonitor(monitoringTime);  // reset the monitor for the next observation
    return make_tuple(observedGlobals, observedNodes, observedEdges,
                      observedTrafficEntries, observedRewards, observedInfo);
}

}   // namespace ns3