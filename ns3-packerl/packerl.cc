#include <iostream>
#include <stdexcept>
#include <memory>

#include "packerl.h"
#include "applications/disposable-bulk-send-application.h"
#include "applications/disposable-onoff-application.h"
#include "odd-routing/odd-node.h"
#include "odd-routing/odd-routing.h"
#include "odd-routing/odd-routing-helper.h"


using std::make_pair, std::vector, std::stoul, std::stringstream, std::distance;

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("PackerlEnv");

PackerlEnv::PackerlEnv(SimParameters simParameters)
    : Ns3AIRL<PackerlEnvStruct, PackerlActStruct>(simParameters.memblockKey),
      m_simParameters(simParameters),
      m_ipv4Block(10 * 255 * 255),
      m_tcpSinkHelper(PacketSinkHelper("ns3::TcpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), m_tcpSinkPort))),
      m_udpSinkHelper(PacketSinkHelper("ns3::UdpSocketFactory", InetSocketAddress(Ipv4Address::GetAny(), m_udpSinkPort)))
{
    // Global configuration
    Config::SetDefault("ns3::OddRouting::RespondToInterfaceEvents", BooleanValue(false));
    Config::SetDefault("ns3::OddRouting::SampleOnDemand", BooleanValue(m_simParameters.probabilisticRouting));

    // Configure TCP applications
    uint32_t tcpSegmentSize = m_simParameters.mtu - m_simParameters.ipv4HeaderSize - m_simParameters.tcpHeaderSize;
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue (tcpSegmentSize));
    Config::SetDefault("ns3::TcpSocketBase::Sack", BooleanValue(m_simParameters.useTcpSack));
    Config::SetDefault("ns3::DisposableBulkSendApplication::SendSize", UintegerValue(tcpSegmentSize));

    // Configure UDP applications
    uint32_t udpPacketSize = m_simParameters.mtu - m_simParameters.ipv4HeaderSize - m_simParameters.udpHeaderSize;
    Config::SetDefault("ns3::DisposableOnOffApplication::PacketSize", UintegerValue (udpPacketSize));
    Config::SetDefault("ns3::DisposableOnOffApplication::OnTime", StringValue("ns3::ConstantRandomVariable[Constant=3600]"));  // TODO: make max_time
    Config::SetDefault("ns3::DisposableOnOffApplication::OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0.0]"));

    // Setup
    m_allNodes = NodeContainer();
    m_p2pHelper.DisableFlowControl();  // TODO include Traffic Control via Queueing Disciplines into our setup at some later point?
    m_inetHelper = InternetStackHelper();
    OddRoutingHelper oddRoutingHelper;
    m_inetHelper.SetRoutingHelper(oddRoutingHelper);
    m_routingLabelMap = oddRoutingHelper.GetOddLabelMap();
    m_packerlMonitor = CreateObject<PackerlMonitor>();
    m_packerlMonitor->SetRoutingLabelMap(m_routingLabelMap);
    m_packerlMonitor->SetCapLinkUtilization(m_simParameters.capLinkUtilization);
    m_packerlMonitor->SetRewardParameters(m_simParameters.localRewardSpatialDecay,
                                          m_simParameters.localRewardPathCutoff,
                                          m_simParameters.localRewardCapToBytes);
    m_packerlMonitor->SetBirdseyeObserverId(m_simParameters.birdseyeObserverId);
}

bool
PackerlEnv::createNode(const uint32_t scenarioNodeId)
{
    NS_LOG_FUNCTION(this << scenarioNodeId);

    // Create node
    Ptr<OddNode> curNode = CreateObject<OddNode>();
    if (curNode->GetId() != scenarioNodeId || m_allNodes.GetN() != scenarioNodeId)
    {
        NS_FATAL_ERROR("Node ID mismatch (they have to be created in order)");
    }

    // Install internet stack on node and link traces
    m_inetHelper.Install(curNode);  // also installs routing; LoopbackNetDevice is now at device index 0

    // create UDP packet sink
    Ptr<PacketSink> udpSinkApp = DynamicCast<PacketSink>(m_udpSinkHelper.Install(curNode).Get(0));
    if (udpSinkApp == nullptr)
    {
        NS_LOG_ERROR("could not cast UDP sink to PacketSink (shouldn't happen)");
        return false;
    }
    udpSinkApp->SetStartTime (Seconds(0.0));
    udpSinkApp->SetStopTime (Simulator::GetMaximumSimulationTime());

    // create TCP packet sink
    Ptr<PacketSink> tcpSinkApp = DynamicCast<PacketSink>(m_tcpSinkHelper.Install(curNode).Get(0));
    if (tcpSinkApp == nullptr)
    {
        NS_LOG_ERROR("could not cast TCP sink to PacketSink (shouldn't happen)");
        return false;
    }
    tcpSinkApp->SetStartTime (Seconds(0.0));
    tcpSinkApp->SetStopTime (Simulator::GetMaximumSimulationTime());

    // finally, register node in monitor and add to list of all nodes
    m_packerlMonitor->RegisterNode(curNode, udpSinkApp, tcpSinkApp);
    m_allNodes.Add(curNode);

    return true;
}

bool
PackerlEnv::createEdge(const TopologyEdge& edge)
{
    NS_LOG_FUNCTION(this << edge.fst << edge.snd);
    // get/prepare current edge information
    stringstream ssdr;
    ssdr << edge.datarate << "bps";  // datarate values are given in bits per second (bps)
    string dataRateStr = ssdr.str();
    Ptr<Node> firstNode = m_allNodes.Get(edge.fst);
    Ptr<Node> secondNode = m_allNodes.Get(edge.snd);

    // install connection and thus get net devices.
    m_p2pHelper.SetDeviceAttribute("DataRate", StringValue(dataRateStr));
    m_p2pHelper.SetChannelAttribute("Delay", TimeValue(
            ns3::Time::FromInteger(edge.delay, ns3::Time::MS)));
    NetDeviceContainer curConnectionNetDevices = m_p2pHelper.Install(firstNode, secondNode);

    // create address space string and assign to new net devices
    // NOTE: there are address spaces for up to 245 * 255 * 255 connections.
    stringstream ss;
    ss << m_ipv4Block / (255 * 255)
       << "." << (m_ipv4Block / 255) % 255
       << "." << m_ipv4Block % 255
       << ".0";
    string ipv4AddressSpace = ss.str();
    m_ipv4Block++;

    m_ipv4Helper.SetBase(ipv4AddressSpace.c_str(), "255.255.255.0");
    Ipv4InterfaceContainer ipic = m_ipv4Helper.Assign(curConnectionNetDevices);

    // newly created NetDevice sits at last idx position
    uint32_t fstDevId = curConnectionNetDevices.Get(0)->GetIfIndex();
    uint32_t sndDevId = curConnectionNetDevices.Get(1)->GetIfIndex();
    auto firstND = make_pair(edge.fst, fstDevId);
    auto secondND = make_pair(edge.snd, sndDevId);
    m_deviceIdsPerEdge.emplace(make_pair(edge.fst, edge.snd), make_pair(fstDevId, sndDevId));  // one direction
    m_deviceIdsPerEdge.emplace(make_pair(edge.snd, edge.fst), make_pair(sndDevId, fstDevId));  // other direction

    // Get Tx buffer capacity via bandwidth-RTT product (BDP),
    // dividing by 1000 to scale datarate to milliseconds and by 8 to get bytes from bits
    uint32_t txQueueCapacity = static_cast<uint32_t>((m_simParameters.bufferSizeBdp * 2.0 * edge.delay * edge.datarate) / (1000.0 * 8.0));
    if (txQueueCapacity < 1500)  // minimum buffer size
    {
        txQueueCapacity = 1500;
    }
    Ptr<PointToPointNetDevice> firstDev = DynamicCast<PointToPointNetDevice>(firstNode->GetDevice(fstDevId));
    if (firstDev == nullptr)
    {
        NS_LOG_ERROR("nullptr device in node/devId" << firstND << " (should not happen)");
    }
    // Set buffer limits according to bandwidth-delay product (BDP)
    firstDev->GetQueue()->SetMaxSize(QueueSize(QueueSizeUnit::BYTES, txQueueCapacity));

    Ptr<PointToPointNetDevice> secondDev = DynamicCast<PointToPointNetDevice>(secondNode->GetDevice(sndDevId));
    if (secondDev == nullptr)
    {
        NS_LOG_ERROR("nullptr device in node/devId" << secondND << " (should not happen)");
    }
    secondDev->GetQueue()->SetMaxSize(QueueSize(QueueSizeUnit::BYTES, txQueueCapacity));

    // Create and hook up two directed monitoredEdges in the monitor, using the incident nodes and devices
    m_packerlMonitor->RegisterEdges(firstDev, firstND, secondDev, secondND, edge.datarate, edge.delay, txQueueCapacity);

    // We'll need to update the sink applications of the two incident nodes so that they are bound to the new devices
    // UDP sinks: first app of respective nodes
    Ptr<PacketSink> fstUdpSink = DynamicCast<PacketSink>(firstNode->GetApplication(0));
    if (fstUdpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast UDP sink to PacketSink (shouldn't happen)");
        return false;
    }
    fstUdpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_udpSinkPort)));

    Ptr<PacketSink> sndUdpSink = DynamicCast<PacketSink>(secondNode->GetApplication(0));
    if (sndUdpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast UDP sink to PacketSink (shouldn't happen)");
        return false;
    }
    sndUdpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_udpSinkPort)));

    // TCP sinks: second app of respective nodes
    Ptr<PacketSink> fstTcpSink = DynamicCast<PacketSink>(firstNode->GetApplication(1));
    if (fstTcpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast TCP sink to PacketSink (shouldn't happen)");
        return false;
    }
    fstTcpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_tcpSinkPort)));

    Ptr<PacketSink> sndTcpSink = DynamicCast<PacketSink>(secondNode->GetApplication(1));
    if (sndTcpSink == nullptr)
    {
        NS_LOG_ERROR("could not cast TCP sink to PacketSink (shouldn't happen)");
        return false;
    }
    sndTcpSink->SetAttribute("Local", AddressValue(InetSocketAddress(Ipv4Address::GetAny(), m_tcpSinkPort)));

    // DONE
    NS_LOG_LOGIC("created edge: (dev: " << fstDevId << ", ip: " << ipic.GetAddress(0) << ") [" << edge.fst << " <-> "
                            << edge.snd << "] (devId=" << sndDevId << ", IP=" << ipic.GetAddress(1) << ") @ datarate(bps)="
                            << edge.datarate << ", delay(ms)=" << edge.delay << ", queuesize(B)=" << txQueueCapacity);
    return true;
}

bool
PackerlEnv::readAndInstallNetworkTopology()
{
    NS_LOG_FUNCTION(this);

    bool nodesReady = false;
    bool edgesReady = false;
    bool topologyReady = nodesReady && edgesReady;
    PackerlEnvStruct *envInfo;

    while (! topologyReady)
    {
        // get data from shared memory
        envInfo = EnvGetterCond();
        bool topologyAvailableData = envInfo->topologyAvailable;
        bool allEdgesSentData = envInfo->allTopologyEdgesSent;
        uint32_t numNodesData = envInfo->numTopologyNodes;
        uint32_t numEdgesAvailableData = envInfo->numTopologyEdgesAvailable;  // needed because not all fields in edgeData might be initialized
        TopologyEdge *edgesData = envInfo->topologyEdges;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool topologyAvailable = topologyAvailableData;
        bool allEdgesSent = allEdgesSentData;
        uint32_t numNodes = numNodesData;
        uint32_t numEdgesAvailable = numEdgesAvailableData;
        vector<TopologyEdge> edges;
        edges.reserve(numEdgesAvailable);
        for (size_t i = 0; i < numEdgesAvailable; ++i)
        {
            edges[i] = edgesData[i];
        }
        GetCompleted();

        // set topologyAvailable to false, so that the python side can send more topology data if there is more
        envInfo = EnvSetterCond();
        envInfo->topologyAvailable = false;
        SetCompleted();

        // if obtained data is valid, use it to initialize the topology
        if (topologyAvailable)
        {
            if (! nodesReady)
            {
                // caution: if, later, there will be configuration for nodes, we can't set up all at once,
                // and we might have to first receive all data before creating edges
                for (uint32_t i = 0; i < numNodes; ++i)
                {
                    if (! createNode(i))
                    {
                        NS_LOG_ERROR("could not create node " << i);
                        return false;
                    }
                }
                NS_LOG_LOGIC("created nodes and installed internet stack on them");
                nodesReady = true;
            }

            if (! edgesReady)
            {
                for (size_t i = 0; i < numEdgesAvailable; ++i)
                {
                    if (! createEdge(edges[i]))
                    {
                        NS_LOG_ERROR("could not create edge " << i);
                        return false;
                    }
                }
                if (allEdgesSent)
                {
                    edgesReady = true;  // edges are ready once we've installed all new ones and no more will be sent
                    NS_LOG_LOGIC("created edges (connections between nodes)");
                }
            }
        }
        topologyReady = nodesReady && edgesReady;
    }

    NS_LOG_FUNCTION(this << "END");
    // print all entries of m_deviceIdsPerEdge (it's an unordered map from pair<uint32_t, uint32_t> to pair<uint32_t, uint32_t>)
    for (auto const& [edge, devIds] : m_deviceIdsPerEdge)
    {
        NS_LOG_LOGIC("device ids for edge " << edge.first << "->" << edge.second
                     << ": " << devIds.first << "->" << devIds.second);
    }

    // TODO: when nodes/edges are added/removed later, we will need to re-register them in the label map
    m_routingLabelMap->registerNodes(m_allNodes);  // call this only once after all edges have been created
    return true;
}

bool
PackerlEnv::readAndSetGlobalRewardWeights()
{
    NS_LOG_FUNCTION(this);

    bool globalRewardsSet = false;
    PackerlEnvStruct *envInfo;

    while (! globalRewardsSet)
    {
        NS_LOG_DEBUG("not all rewards set yet");
        // get data
        envInfo = EnvGetterCond();
        bool globalRewardWeightsAvailableData = envInfo->globalRewardWeightsAvailable;
        GlobalRewardWeights globalRewardWeightsData = envInfo->globalRewardWeights;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool globalRewardWeightsAvailable = globalRewardWeightsAvailableData;
        GlobalRewardWeights globalRewardWeights = globalRewardWeightsData;
        GetCompleted();

        // set globalRewardWeightsAvailable to false, so that the python side can send data if there is more
        envInfo = EnvSetterCond();
        envInfo->globalRewardWeightsAvailable = false;
        SetCompleted();

        if (globalRewardWeightsAvailable)
        {
            m_packerlMonitor->SetGlobalRewardWeights(globalRewardWeights);
            globalRewardsSet = true;
        }
    }

    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::createApplicationForFlow(const TrafficFlowArrival& flowArrival)
{
    // first, get nodes and source port
    Ptr<OddNode> srcNode = DynamicCast<OddNode>(m_allNodes.Get(flowArrival.src));
    if (srcNode == nullptr)
    {
        NS_LOG_ERROR("could not cast source node to OddNode (shouldn't happen since they're all OddNodes)");
        return false;
    }
    Ptr<OddNode> dstNode = DynamicCast<OddNode>(m_allNodes.Get(flowArrival.dst));
    if (dstNode == nullptr)
    {
        NS_LOG_ERROR("could not cast destination node to OddNode (shouldn't happen since they're all OddNodes)");
        return false;
    }

    uint16_t srcPort = srcNode->reservePort();
    NS_LOG_FUNCTION(this << flowArrival.src << flowArrival.dst << srcPort);
    if (srcPort == 0)  // for OddNodes, return value 0 signifies there are no free ports
    {
        NS_LOG_ERROR("could not reserve free port for source node " << flowArrival.src);
        return false;
    }

    // create application according to flow arrival type. Note: some app attributes are set in the class initializer list
    // We use custom variants of the applications that allow for disposal upon completion, for sending smaller
    // last packets in the case of UDP applications, and for registering retransmits in the case of TCP applications.
    Ptr<DisposableApplication> app;
    uint16_t dstPort;
    if (flowArrival.isTCP)  // TCP: DisposableBulkSendApplication
    {
        app = CreateObject<DisposableBulkSendApplication>();
        m_packerlMonitor->RegisterTcpSenderApp(app);
        // app->TraceConnect("ReadyToDispose", std::to_string(flowArrival.src), MakeCallback(&PackerlEnv::removeApplication, this));  // TODO if working on this, also disconnect monitoring
        dstPort = m_tcpSinkPort;
    }
    else  // UDP: DisposableOnOffApplication
    {
        app = CreateObject<DisposableOnOffApplication>();
        app->SetAttribute("DataRate", DataRateValue(DataRate(flowArrival.datarate)));
        m_packerlMonitor->RegisterUdpSenderApp(app);
        // app->TraceConnect("ReadyToDispose", std::to_string(flowArrival.src), MakeCallback(&PackerlEnv::removeApplication, this));    // TODO if working on this, also disconnect monitoring
        dstPort = m_udpSinkPort;
    }
    app->SetAttribute("MaxBytes", UintegerValue(flowArrival.amount));

    // NOTE: getDefaultIpv4Address() is a function specific to OddNode returning the first non-loopback address
    AddressValue sourceAddress(InetSocketAddress(srcNode->getDefaultIpv4Address(), srcPort));
    app->SetAttribute("Local", sourceAddress);
    AddressValue remoteAddress(InetSocketAddress(dstNode->getDefaultIpv4Address(), dstPort));
    app->SetAttribute("Remote", remoteAddress);

    // Set start time relative to current simulation time
    double relativeTime = 0.001 * (flowArrival.t - Simulator::Now().GetMilliSeconds());
    app->SetStartTime(Seconds(relativeTime));
    app->SetStopTime (Simulator::GetMaximumSimulationTime());

    // Finally, add the sender application to the source node
    srcNode->AddApplication(app);
    NS_LOG_LOGIC("created flowArrival " << flowArrival.src << " -> " << flowArrival.dst << " at "
                              << flowArrival.t << "ms (relative: " << relativeTime << "s): "
                              << flowArrival.amount << " bytes, " << flowArrival.datarate << " bps, "
                              << (flowArrival.isTCP ? "TCP" : "UDP"));
    return true;
}

void
PackerlEnv::removeApplication(string context, Ptr<DisposableApplication> app)
{
    NS_LOG_FUNCTION(this << context << app);
    uint32_t nodeId = static_cast<uint32_t>(stoul(context));  // context is the node id
    Ptr<OddNode> curNode = DynamicCast<OddNode>(m_allNodes.Get(nodeId));
    if (curNode == nullptr)
    {
        NS_FATAL_ERROR("could not cast node to OddNode (shouldn't happen since they're all OddNodes)");
    }
    AddressValue appLocalAddress;
    app->GetAttribute("Local", appLocalAddress);
    uint16_t appLocalPort = InetSocketAddress::ConvertFrom(appLocalAddress.Get()).GetPort();
    if (! curNode->RemoveApplication(app, appLocalPort))
    {
        NS_FATAL_ERROR("could not remove application in node " << nodeId);
    }
    NS_LOG_FUNCTION(this << context << app << "END");
}

bool
PackerlEnv::executeLinkFailure(const LinkFailure& linkFailure)
{
    NS_LOG_FUNCTION(this << linkFailure.t << "s: " << linkFailure.fst << " -> " << linkFailure.snd);
    uint32_t fst = linkFailure.fst;
    uint32_t snd = linkFailure.snd;
    auto it = m_deviceIdsPerEdge.find(make_pair(fst, snd));
    if (it == m_deviceIdsPerEdge.end())
    {
        NS_LOG_ERROR("could not find device ids for edge " << fst << ", " << snd);
        return false;
    }
    auto [fstDevId, sndDevId] = it->second;
    Ptr<OddNode> fstNode = DynamicCast<OddNode>(m_allNodes.Get(fst));
    Ptr<OddNode> sndNode = DynamicCast<OddNode>(m_allNodes.Get(snd));
    if (fstNode == nullptr || sndNode == nullptr)
    {
        NS_LOG_ERROR("could not cast nodes to OddNode");
        return false;
    }
    Ptr<PointToPointNetDevice> fstDev = DynamicCast<PointToPointNetDevice>(fstNode->GetDevice(fstDevId));
    Ptr<PointToPointNetDevice> sndDev = DynamicCast<PointToPointNetDevice>(sndNode->GetDevice(sndDevId));
    if (fstDev == nullptr || sndDev == nullptr)
    {
        NS_LOG_ERROR("could not cast devices to PointToPointNetDevice");
        return false;
    }
    // we deactivate the entire link by setting the error model on incident network devices
    // to a rate error model with 100% error rate
    Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
    em->SetAttribute("ErrorRate", DoubleValue(1.0));
    fstDev->SetAttribute("ReceiveErrorModel", PointerValue(em));
    sndDev->SetAttribute("ReceiveErrorModel", PointerValue(em));

    // register link down in monitoring edges
    m_packerlMonitor->RegisterLinkDown(fst, snd);

    return true;
}

bool
PackerlEnv::readAndInstallEvents()
{
    NS_LOG_FUNCTION(this);

    bool allEventsRead = false;
    PackerlEnvStruct *envInfo;
    while (!allEventsRead) {
        // get data
        envInfo = EnvGetterCond();
        bool upcomingEventsAvailableData = envInfo->upcomingEventsAvailable;
        bool allUpcomingEventsSentData = envInfo->allUpcomingEventsSent;
        uint32_t numUpcomingEventsAvailableData = envInfo->numUpcomingEventsAvailable;
        Event *upcomingEventsData = envInfo->upcomingEvents;
        uint32_t *upcomingEventTypesData = envInfo->upcomingEventTypes;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool upcomingEventsAvailable = upcomingEventsAvailableData;
        bool allUpcomingEventsSent = allUpcomingEventsSentData;
        uint32_t numUpcomingEventsAvailable = numUpcomingEventsAvailableData;
        vector<Event> upcomingEvents;
        vector<uint32_t> upcomingEventTypes;
        upcomingEvents.reserve(numUpcomingEventsAvailable);
        upcomingEventTypes.reserve(numUpcomingEventsAvailable);
        for (size_t i = 0; i < numUpcomingEventsAvailable; ++i)
        {
            upcomingEvents[i] = upcomingEventsData[i];
            upcomingEventTypes[i] = upcomingEventTypesData[i];
        }
        GetCompleted();

        // set available to false, so that the python side can send more data if there is more
        envInfo = EnvSetterCond();
        envInfo->upcomingEventsAvailable = false;
        SetCompleted();

        // if obtained data is valid, use it to initialize the topology
        if (upcomingEventsAvailable) {
            for (size_t i = 0; i < numUpcomingEventsAvailable; ++i) {
                Event e = upcomingEvents[i];
                uint32_t eventType = upcomingEventTypes[i];
                if (eventType == 0)  // TrafficFlowArrival
                {
                    TrafficFlowArrival flowArrival = e.trafficFlowArrival;
                    if (! createApplicationForFlow(flowArrival))
                    {
                        NS_LOG_ERROR("could not create application from flow arrival");
                        return false;
                    }
                }
                else if (eventType == 1)  // LinkFailure
                {
                    LinkFailure linkFailure = e.linkFailure;
                    double relativeTime = 0.001 * (linkFailure.t - Simulator::Now().GetMilliSeconds());
                    Simulator::Schedule(Seconds(relativeTime), &PackerlEnv::executeLinkFailure, this, linkFailure);
                }
                else
                {
                    NS_LOG_ERROR("Unsupported event type " << eventType);
                }
            }
            NS_LOG_LOGIC("Read and installed " << numUpcomingEventsAvailable << " events");
            if (allUpcomingEventsSent) {
                allEventsRead = true;  // events are ready once we've installed all new ones and no more will be sent
            }
        }
    }
    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::readActions(OutInterfaceDistMap& oifDistPerDestinationPerNode,
                        OutInterfaceMap& oifPerDestinationPerNode,
                        unordered_map<uint32_t, float>& agentDelaysPerNode)
{
    NS_LOG_FUNCTION(this);

    PackerlActStruct *actInfo;
    bool allActionsRead = false;
    bool allAgentDelaysRead = false;
    while (! allActionsRead || ! allAgentDelaysRead) {

        // get data
        actInfo = ActionGetterCond();

        bool actionsAvailableData = actInfo->actionsAvailable;
        bool allActionsSentData = actInfo->allActionsSent;
        uint32_t numRoutingActionsAvailableData = actInfo->numRoutingActionsAvailable;  // needed because not all fields might be initialized
        RoutingActionComponent *routingActionsData = actInfo->routingActions;

        bool agentDelaysAvailableData = actInfo->routingAgentDelaysAvailable;
        bool allAgentDelaysSentData = actInfo->allRoutingAgentDelaysSent;
        uint32_t numRoutingAgentDelaysAvailableData = actInfo->numRoutingAgentDelaysAvailable;  // needed because not all fields might be initialized
        RoutingAgentDelay *agentDelaysData = actInfo->routingAgentDelays;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool actionsAvailable = actionsAvailableData;
        bool allActionsSent = allActionsSentData;
        uint32_t numRoutingActionsAvailable = numRoutingActionsAvailableData;
        vector<RoutingActionComponent> routingActions;
        routingActions.reserve(numRoutingActionsAvailable);
        for (size_t i = 0; i < numRoutingActionsAvailable; ++i)
        {
            routingActions[i] = routingActionsData[i];
        }
        bool agentDelaysAvailable = agentDelaysAvailableData;
        bool allAgentDelaysSent = allAgentDelaysSentData;
        uint32_t numRoutingAgentDelaysAvailable = numRoutingAgentDelaysAvailableData;
        vector<RoutingAgentDelay> agentDelays;
        agentDelays.reserve(numRoutingAgentDelaysAvailable);
        for (size_t i = 0; i < numRoutingAgentDelaysAvailable; ++i)
        {
            agentDelays[i] = agentDelaysData[i];
        }
        GetCompleted();

        // set available to false, so that the python side can send more data if there is more
        actInfo = ActionSetterCond();
        actInfo->actionsAvailable = false;
        actInfo->routingAgentDelaysAvailable = false;
        SetCompleted();

        // fill in action information
        if (actionsAvailable) {
            for (size_t i = 0; i < numRoutingActionsAvailable; ++i) {
                RoutingActionComponent rac = routingActions[i];
                uint32_t edgeSrc = rac.edgeSrc;
                uint32_t edgeDst = rac.edgeDst;
                uint32_t trafficDst = rac.trafficDst;
                float value = rac.value;
                uint32_t nextHopDevId = m_deviceIdsPerEdge[make_pair(edgeSrc, edgeDst)].first;
                NS_LOG_LOGIC(this << " routing action: (" << edgeSrc << ", " << edgeDst
                                  << "), trafficDst: " << trafficDst
                                  << ", nextHopDevId: " << nextHopDevId
                                  <<  ", value: " << value);
                if (m_simParameters.probabilisticRouting)  // probabilistic routing -> assign probability
                {
                    oifDistPerDestinationPerNode[edgeSrc][trafficDst][nextHopDevId] = value;
                }
                else if (value > 0.)  // deterministic routing and this edge is the maximum edge
                {
                    oifPerDestinationPerNode[edgeSrc][trafficDst] = nextHopDevId;
                }
            }
            if (allActionsSent) {
                allActionsRead = true;  // ready once we've read all new ones and no more will be sent
            }
        }
        if (agentDelaysAvailable) {
            for (size_t i = 0; i < numRoutingAgentDelaysAvailable; ++i) {
                RoutingAgentDelay rad = agentDelays[i];
                uint32_t nodeId = rad.nodeId;
                float delay = rad.delay;
                NS_LOG_LOGIC(this << " routing agent delay: node " << nodeId << ", delay: " << delay);
                agentDelaysPerNode.emplace(nodeId, delay);
            }
            if (allAgentDelaysSent) {
                allAgentDelaysRead = true;  // ready once we've read all new ones and no more will be sent
            }
        }
    }
    return true;
}

bool
PackerlEnv::installProbabilisticActionForNode(Ptr<OddNode> node,
                                              const unordered_map<uint32_t, map<uint32_t, double>>& outInterfaceDistMapForV)
{
    NS_LOG_FUNCTION(this << node->GetId());
    for (auto const& [dst, nextHopDist] : outInterfaceDistMapForV)
    {
        for (auto const& [nextHopId, p] : nextHopDist)
        {
            NS_LOG_LOGIC("routing @ " << node->GetId() << " to " << dst << " via nextHopId " << nextHopId << ": p=" << p);
        }
    }
    node->setOifDistPerDestination(outInterfaceDistMapForV);
    return true;
}

bool
PackerlEnv::installDeterministicActionForNode(Ptr<OddNode> node,
                                              const unordered_map<uint32_t, uint32_t>& outInterfaceMapForV)
{
    NS_LOG_FUNCTION(this << node->GetId());
    for (auto const& [dst, nextHopId] : outInterfaceMapForV)
    {
        NS_LOG_LOGIC("routing @ " << node->GetId() << " to " << dst << ": nextHopId = " << nextHopId);
    }
    node->setOifPerDestination(outInterfaceMapForV);
    return true;
}

bool
PackerlEnv::readAndInstallActions()
{
    NS_LOG_FUNCTION(this);

    // read action information from shared memory
    OutInterfaceDistMap oifDistPerDestinationPerNode;
    OutInterfaceMap oifPerDestinationPerNode;
    unordered_map<uint32_t, float> agentDelaysPerNode;
    if (! readActions(oifDistPerDestinationPerNode, oifPerDestinationPerNode, agentDelaysPerNode))
    {
        NS_LOG_ERROR("could not read actions from shared memory");
        return false;
    }
    NS_LOG_LOGIC("read OddRoutingAction from shm");

    // install routing decisions in nodes
    for (auto const& [v, delay] : agentDelaysPerNode)
    {
        Ptr<OddNode> curNode = DynamicCast<OddNode>(m_allNodes.Get(v));
        if (curNode == nullptr)
        {
            NS_LOG_ERROR("Node is not an OddNode: " << v);
            return false;
        }
        if (m_simParameters.probabilisticRouting)
        {
            OutInterfaceDistMap::const_iterator itOifDist = oifDistPerDestinationPerNode.find(v);
            if (itOifDist == oifDistPerDestinationPerNode.end())
            {
                NS_LOG_ERROR("could not find oifDist for OddNode " << v);
                return false;
            }
            // install right away if delay is zero, otherwise schedule installation after delay
            if (delay == 0.)
            {
                if (! installProbabilisticActionForNode(curNode, itOifDist->second))
                {
                    NS_LOG_ERROR("could not install probabilistic action for OddNode " << v);
                    return false;
                }
            }
            else
            {
                Simulator::Schedule(MilliSeconds(delay), &PackerlEnv::installProbabilisticActionForNode, this,
                                    curNode, itOifDist->second);
                NS_LOG_LOGIC("scheduled action install (dist) for OddNode " << v << " after delay " << delay << "s");
            }
        }
        else
        {
            OutInterfaceMap::const_iterator itOif = oifPerDestinationPerNode.find(v);
            if (itOif == oifPerDestinationPerNode.end())
            {
                NS_LOG_ERROR("could not find oif for OddNode " << v);
                return false;
            }
            // install right away if delay is zero, otherwise schedule installation after delay
            if (delay == 0.)
            {
                if (! installDeterministicActionForNode(curNode, itOif->second))
                {
                    NS_LOG_ERROR("could not install deterministic action for OddNode " << v);
                    return false;
                }
            }
            else
            {
                Simulator::Schedule(MilliSeconds(delay), &PackerlEnv::installDeterministicActionForNode, this,
                                    curNode, itOif->second);
                NS_LOG_LOGIC("scheduled action install (det) for OddNode " << v << " after delay " << delay << "s");
            }
        }
    }
    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::readAndInstallCommunicationDelays()
{
    NS_LOG_FUNCTION(this);

    U32ToU32ToDMap communicationDelays;
    uint32_t centralNode;

    bool allCommunicationDelaysRead = false;
    PackerlEnvStruct *envInfo;

    while (!allCommunicationDelaysRead) {

        // get data
        envInfo = EnvGetterCond();
        bool communicationDelaysAvailableData = envInfo->communicationDelaysAvailable;
        bool allCommunicationDelaysSentData = envInfo->allCommunicationDelaysSent;
        uint32_t numCommunicationDelaysAvailableData = envInfo->numCommunicationDelaysAvailable;
        CommunicationDelay *communicationDelaysData = envInfo->communicationDelays;
        uint32_t centralNodeData = envInfo->centralNode;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool communicationDelaysAvailable = communicationDelaysAvailableData;
        bool allCommunicationDelaysSent = allCommunicationDelaysSentData;
        uint32_t numCommunicationDelaysAvailable = numCommunicationDelaysAvailableData;
        vector<CommunicationDelay> communicationDelaysVec;
        communicationDelaysVec.reserve(numCommunicationDelaysAvailable);
        for (size_t i = 0; i < numCommunicationDelaysAvailable; ++i)
        {
            communicationDelaysVec[i] = communicationDelaysData[i];
        }
        centralNode = centralNodeData;
        GetCompleted();

        // set available to false, so that the python side can send more data if there is more
        envInfo = EnvSetterCond();
        envInfo->communicationDelaysAvailable = false;
        SetCompleted();

        if (communicationDelaysAvailable) {
            for (size_t i = 0; i < numCommunicationDelaysAvailable; ++i) {
                const CommunicationDelay commDelay = communicationDelaysVec[i];
                if (communicationDelays.find(commDelay.src) == communicationDelays.end())
                {
                    communicationDelays[commDelay.src] = U32ToDMap();
                }
                communicationDelays[commDelay.src][commDelay.dst] = commDelay.delay;
            }
            NS_LOG_LOGIC("Read and installed " << numCommunicationDelaysAvailable << " events");
            if (allCommunicationDelaysSent) {
                allCommunicationDelaysRead = true;  // communication delays are ready once we've installed all new ones and no more will be sent
            }
        }
    }
    m_packerlMonitor->SetCommunicationDelaysAndCentralNode(communicationDelays, centralNode);

    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::getDone()
{
    NS_LOG_FUNCTION(this);

    bool doneRead = false;
    bool done;
    PackerlEnvStruct *envInfo;

    while (! doneRead) {
        envInfo = EnvGetterCond();
        bool doneAvailableData = envInfo->doneAvailable;
        bool doneData = envInfo->done;

        // before yielding shm control, copy data to local array (avoid python side overwriting)
        bool doneAvailable = doneAvailableData;
        done = doneData;
        GetCompleted();

        if (doneAvailable) {
            doneRead = true;
        }

        // set doneAvailable to false, so that the python side can send it again later
        envInfo = EnvSetterCond();
        envInfo->doneAvailable = false;
        SetCompleted();

    }
    NS_LOG_FUNCTION(this << "END");
    return done;
}

bool
PackerlEnv::runSimStep()
{
    NS_LOG_FUNCTION(this);
    try
    {
        NS_LOG_LOGIC("running sim step...");
        Simulator::Stop(MicroSeconds(m_simParameters.simStepDuration * 1000));  // convert to microseconds to support fractional values for simStepDuration
        Simulator::Run();
        NS_LOG_LOGIC("Simulation time is " << Simulator::Now().GetMilliSeconds() << "ms");
    }
    catch (const std::exception& e)
    {
        NS_LOG_ERROR(e.what());
        return false;
    }
    NS_LOG_FUNCTION(this << "END");
    return true;
}

bool
PackerlEnv::observeNetwork()
{
    NS_LOG_FUNCTION(this);

    // obtain network observation
    NetworkObservation observation = m_packerlMonitor->ObserveNetwork();

    // prepare observation for sending
    bool observedInfoSent = false;
    bool allObservedGlobalsSent = false;
    bool allObservedNodesSent = false;
    bool allObservedEdgesSent = false;
    bool allObservedTrafficEntriesSent = false;
    bool allObservedRewardsSent = false;
    bool observationSent = observedInfoSent
                          && allObservedGlobalsSent
                          && allObservedNodesSent
                          && allObservedEdgesSent
                          && allObservedTrafficEntriesSent
                          && allObservedRewardsSent;
    PackerlEnvStruct *envInfo;
    auto const& [observedGlobals, observedNodes, observedEdges,
                 observedTrafficEntries, observedRewards, observedInfo] = observation;

    auto observedGlobalIt = observedGlobals.cbegin();
    const auto observedGlobalsEnd = observedGlobals.cend();
    auto observedNodeIt = observedNodes.cbegin();
    const auto observedNodesEnd = observedNodes.cend();
    auto observedEdgeIt = observedEdges.cbegin();
    const auto observedEdgesEnd = observedEdges.cend();
    auto observedTrafficEntryIt = observedTrafficEntries.cbegin();
    const auto observedTrafficEntriesEnd = observedTrafficEntries.cend();
    auto observedRewardIt = observedRewards.cbegin();
    const auto observedRewardsEnd = observedRewards.cend();

    // send observation, in slices if necessary: Stay in the shm loop until all components have been sent
    while (! observationSent)
    {
        envInfo = EnvSetterCond();
        // if there is still observation data in the shared memory, the python side has not yet consumed the last data
        if (! envInfo->observationAvailable)
        {
            // observed info
            if (! observedInfoSent)
            {
                envInfo->observedInfo = observedInfo;
                observedInfoSent = true;
            }

            // observed globals
            if (! allObservedGlobalsSent)
            {
                uint32_t numObservedGlobalsToBeSent;
                bool allObservedGlobalsSendable;
                if (distance(observedGlobalIt, observedGlobalsEnd) <= MAX_ELEMS)
                {
                    numObservedGlobalsToBeSent = distance(observedGlobalIt, observedGlobalsEnd);
                    allObservedGlobalsSendable = true;
                }
                else
                {
                    numObservedGlobalsToBeSent = MAX_ELEMS;
                    allObservedGlobalsSendable = false;
                }
                for (size_t i = 0; i < numObservedGlobalsToBeSent; ++i, ++observedGlobalIt)
                {
                    NS_LOG_LOGIC("sending observedGlobal for observerId=" << observedGlobalIt->observerId);
                    envInfo->observedGlobals[i] = *observedGlobalIt;
                }
                envInfo->numAvailableObservedGlobals = numObservedGlobalsToBeSent;
                envInfo->allObservedGlobalsSent = allObservedGlobalsSendable;
                if (allObservedGlobalsSendable)
                {
                    allObservedGlobalsSent = true;
                }
            }

            // observed nodes
            if (! allObservedNodesSent)
            {
                uint32_t numObservedNodesToBeSent;
                bool allObservedNodesSendable;
                if (distance(observedNodeIt, observedNodesEnd) <= MAX_ELEMS)
                {
                    numObservedNodesToBeSent = distance(observedNodeIt, observedNodesEnd);
                    allObservedNodesSendable = true;
                }
                else
                {
                    numObservedNodesToBeSent = MAX_ELEMS;
                    allObservedNodesSendable = false;
                }
                for (size_t i = 0; i < numObservedNodesToBeSent; ++i, ++observedNodeIt)
                {
                    NS_LOG_LOGIC("sending observedNode"
                                 << ", observerId=" << observedNodeIt->observerId
                                 << ", nodeId=" << observedNodeIt->nodeId);
                    envInfo->observedNodes[i] = *observedNodeIt;
                }
                envInfo->numAvailableObservedNodes = numObservedNodesToBeSent;
                envInfo->allObservedNodesSent = allObservedNodesSendable;
                if (allObservedNodesSendable)
                {
                    allObservedNodesSent = true;
                }
            }

            // observed edges
            if (! allObservedEdgesSent)
            {
                uint32_t numObservedEdgesToBeSent;
                bool allObservedEdgesSendable;
                if (distance(observedEdgeIt, observedEdgesEnd) <= MAX_ELEMS)
                {
                    numObservedEdgesToBeSent = distance(observedEdgeIt, observedEdgesEnd);
                    allObservedEdgesSendable = true;
                }
                else
                {
                    numObservedEdgesToBeSent = MAX_ELEMS;
                    allObservedEdgesSendable = false;
                }
                for (size_t i = 0; i < numObservedEdgesToBeSent; ++i, ++observedEdgeIt)
                {
                    NS_LOG_LOGIC("sending observedEdge"
                                 << ", observerId=" << observedEdgeIt->observerId
                                 << ", src=" << observedEdgeIt->src
                                 << ", dst=" << observedEdgeIt->dst);
                    envInfo->observedEdges[i] = *observedEdgeIt;
                }
                envInfo->numAvailableObservedEdges = numObservedEdgesToBeSent;
                envInfo->allObservedEdgesSent = allObservedEdgesSendable;
                if (allObservedEdgesSendable)
                {
                    allObservedEdgesSent = true;
                }
            }

            // observed matrix entries
            if (! allObservedTrafficEntriesSent)
            {
                uint32_t numObservedTrafficEntriesToBeSent;
                bool allObservedTrafficEntriesSendable;
                if (distance(observedTrafficEntryIt, observedTrafficEntriesEnd) <= MAX_ELEMS)
                {
                    numObservedTrafficEntriesToBeSent = distance(observedTrafficEntryIt, observedTrafficEntriesEnd);
                    allObservedTrafficEntriesSendable = true;
                }
                else
                {
                    numObservedTrafficEntriesToBeSent = MAX_ELEMS;
                    allObservedTrafficEntriesSendable = false;
                }
                for (size_t i = 0; i < numObservedTrafficEntriesToBeSent; ++i, ++observedTrafficEntryIt)
                {
                    envInfo->observedTrafficEntries[i] = *observedTrafficEntryIt;
                }
                envInfo->numAvailableObservedTrafficEntries = numObservedTrafficEntriesToBeSent;
                envInfo->allObservedTrafficEntriesSent = allObservedTrafficEntriesSendable;
                if (allObservedTrafficEntriesSendable)
                {
                    allObservedTrafficEntriesSent = true;
                }
            }

            // observed rewards
            if (! allObservedRewardsSent)
            {
                uint32_t numObservedRewardsToBeSent;
                bool allObservedRewardsSendable;
                if (distance(observedRewardIt, observedRewardsEnd) <= MAX_ELEMS)
                {
                    numObservedRewardsToBeSent = distance(observedRewardIt, observedRewardsEnd);
                    allObservedRewardsSendable = true;
                }
                else
                {
                    numObservedRewardsToBeSent = MAX_ELEMS;
                    allObservedRewardsSendable = false;
                }
                for (size_t i = 0; i < numObservedRewardsToBeSent; ++i, ++observedRewardIt)
                {
                    NS_LOG_LOGIC("sending observedReward for observerId=" << observedRewardIt->observerId);
                    envInfo->observedRewards[i] = *observedRewardIt;
                }
                envInfo->numAvailableObservedRewards = numObservedRewardsToBeSent;
                envInfo->allObservedRewardsSent = allObservedRewardsSendable;
                if (allObservedRewardsSendable)
                {
                    allObservedRewardsSent = true;
                }
            }
            envInfo->observationAvailable = true;
        }
        SetCompleted();
        observationSent = observedInfoSent
                          && allObservedGlobalsSent
                          && allObservedNodesSent
                          && allObservedEdgesSent
                          && allObservedTrafficEntriesSent
                          && allObservedRewardsSent;  // check if all components have been sent
    }

    // wait for python side to signal that it has fully consumed the data, then yield shm control
    bool observationAvailable;
    do
    {
        envInfo = EnvGetterCond();
        observationAvailable = envInfo->observationAvailable;
        GetCompleted();
    } while (observationAvailable);

    return true;
}

bool
PackerlEnv::runSimLoop()
{
    NS_LOG_FUNCTION(this);

    // PREPARATION: network creation and initial observation
    if (! readAndInstallNetworkTopology()) { return false; }
    if (! readAndSetGlobalRewardWeights()) { return false; }
    if (! readAndInstallCommunicationDelays()) { return false; }
    if (! observeNetwork()) { return false; }

    // MAIN EPISODE LOOP. Termination due to episode length controlled by python side
    // with the 'done' variable; otherwise, only abnormal behaviour terminates this loop on the C++ side.
    bool done = false;
    while (!done)
    {
        if (! readAndInstallEvents()) { return false; }
        if (! readAndInstallActions()) { return false; }
        if (! readAndInstallCommunicationDelays()) { return false; }
        if (! runSimStep()) { return false; }
        if (! observeNetwork()) { return false; }
        done = getDone();
        if (done) { SetFinish(); }
    }
    return true;
}

} // namespace ns3