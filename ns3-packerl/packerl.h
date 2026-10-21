#ifndef NS3_PACKERL_H
#define NS3_PACKERL_H

#include <tuple>

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/traffic-control-module.h"
#include "ns3/ns3-ai-module.h"

#include "utils/shared-structs.h"
#include "utils/sim-parameters.h"
#include "utils/types.h"
#include "applications/disposable-application.h"
#include "monitoring/packerl-monitor.h"
#include "odd-routing/odd-label-map.h"


using std::string, std::unordered_map, std::map;

typedef unordered_map<uint32_t, unordered_map<uint32_t, map<uint32_t, double>>> OutInterfaceDistMap;
typedef unordered_map<uint32_t, unordered_map<uint32_t, uint32_t>> OutInterfaceMap;

namespace ns3
{

/**
 * The PackerlEnv class defines the simulation 'environment' for a reinforcement learning experiment
 * on routing optimization with ns3. It contains several methods to set up an ns3 simulation
 * environment including pre-defined/customizable topologies and applications, run simulation steps
 * of pre-configurable length and monitor the results and effects on the network. It uses a memory
 * block that is shared with a learning loop running in python that provides actions and reads the
 * monitoring, aiming to learn optimal routing behavior within the simulation scenario described
 * here.
 */
class PackerlEnv : public Ns3AIRL<PackerlEnvStruct, PackerlActStruct>
{
  public:
    /**
     * Constructor that sets/initializes necessary values
     */
    PackerlEnv(SimParameters simParameters);

    /**
     * Runs the main simulation loop of an episode. First sets up the network before iteratively
     * installing the actions delivered by the python side, executing a simulation step and
     * providing the newest network measurements back to the python side. Includes extensive anomaly
     * handling that also notify the python side if needed.
     * @return true if the episode terminated without issue, false otherwise.
     */
    bool runSimLoop();

  private:

    /**
     * Creates a network node with given ID.
     * @param i
     * @return true if successful, false otherwise.
     */
    bool createNode(uint32_t i);

    /**
     * Creates a network edge between two nodes.
     * @param edge
     * @return true if successful, false otherwise.
     */
    bool createEdge(const TopologyEdge& edge);

    /**
     * Creates a network topology from shm graph =(edge list and number of nodes).
     * @return true if setup was successful, false otherwise.
     */
    bool readAndInstallNetworkTopology();

    /**
     * Reads the global reward weights from shared memory and sets them in the simulation.
     */
    bool readAndSetGlobalRewardWeights();

    /**
     * Creates an application sending data from the given flow arrival information.
     * @param flowArrival
     * @return true if successful, false otherwise.
     */
    bool createApplicationForFlow(const TrafficFlowArrival& flowArrival);

    /**
     * Removes an application from the simulation.
     */
    void removeApplication(string context, Ptr<DisposableApplication> app);

    /**
     * Deactivates the specified link in the network.
     * @return true if successful, false otherwise.
     */
    bool executeLinkFailure(const LinkFailure& linkFailure);

    /**
     * Consults the shared memory for the newest events (traffic flow arrivals and e.g. link failures) that shall be
     * transferred to the simulation (to the source applications).
     */
    bool readAndInstallEvents();

    /**
     * Reads the actions from shared memory.
     * @param oifDistPerDestinationPerNode output parameter to hold the out interface distribution map
     *                                     per destination per node
     * @param oifPerDestinationPerNode output parameter to hold the out interface map per destination per node
     * @param agentDelaysPerNode output parameter to hold the communication delays per node
     * @return true if successful, false otherwise.
     */
    bool readActions(OutInterfaceDistMap& oifDistPerDestinationPerNode,
                     OutInterfaceMap& oifPerDestinationPerNode,
                     unordered_map<uint32_t, float>& agentDelaysPerNode);

    /**
     * Installs a probabilistic routing action for the given node in the simulation.
     * @param node the node to install the action for
     * @param outInterfaceDistMapForV the out interface distribution map for all destinations v
     * @return true if successful, false otherwise.
     */
    bool installProbabilisticActionForNode(Ptr<OddNode> node,
                                           const unordered_map<uint32_t, map<uint32_t, double>>& outInterfaceDistMapForV);

    /**
     * Installs a deterministic routing action for the given node in the simulation.
     * @param node the node to install the action for
     * @param outInterfaceMapForV the out interface map for all destinations v
     * @return true if successful, false otherwise.
     */
    bool installDeterministicActionForNode(Ptr<OddNode> node,
                                           const unordered_map<uint32_t, uint32_t>& outInterfaceMapForV);

    /**
     * Consults the shared memory for the newest actions that shall be transferred to the simulation,
     * then installs them, including any necessary delay per agent.
     */
    bool readAndInstallActions();

    /**
     * Consults the shared memory for the newest communication delays that shall be transferred to the simulation.
     */
    bool readAndInstallCommunicationDelays();

    /**
     * Consults the shared memory for whether the done signal has been sent by the python side
     * (indicating that the episode is over and simulation should be terminated).
     * @return true if successful, false otherwise.
     */
    bool getDone();

    /**
     * Runs a single simulation step of the episode.
     */
    bool runSimStep();

    /**
     * Measures/monitors the network for metric such as load, queuing times, availability etc.
     * Places this information in the simulation object's member variable.
     * @return true if successful, false otherwise.
     */
    bool observeNetwork();

    // === Member variables ========================================================

    /**
     * Simulation parameters such as timesteps, packet size etc.
     */
    SimParameters m_simParameters;

    /**
     * A helper object to set up point-to-point links between nodes.
     */
    PointToPointHelper m_p2pHelper;

    /**
     * A helper object to set up IP components for nodes.
     */
    Ipv4AddressHelper m_ipv4Helper;

    /**
     * Current ipv4 block for the simulation (used for setting up address spaces between nodes)
     */
    int m_ipv4Block;

    /**
     * Internet stack helper used throughout the simulation
     */
    InternetStackHelper m_inetHelper;

    /**
     * A helper object to create packet sinks for TCP applications.
     */
    PacketSinkHelper m_tcpSinkHelper;

    /**
     * Default port for all TCP sink applications.
     */
    const uint16_t m_tcpSinkPort = 1;

    /**
     * A helper object to create packet sinks for UDP applications.
     */
    PacketSinkHelper m_udpSinkHelper;

    /**
     * Default port for all UDP sink applications.
     */
    const uint16_t m_udpSinkPort = 2;

    /**
     * Holds all currently initialized simulation nodes.
     */
    NodeContainer m_allNodes;

    /**
     * A map from node IDs to the device IDs that span the actual channel between these two nodes.
     * keys: (srcId, dstId); values: (srcDeviceIdOnNode, dstDeviceIdOnNode)
     */
    U32PairToU32PairMap m_deviceIdsPerEdge;

    /**
     * The monitor object that processes events/traces, measures the network and provides monitoring snapshtos back
     * to this environment.
     */
    Ptr<PackerlMonitor> m_packerlMonitor;

    /**
     * A map from IP addresses to node IDs, used for routing.
     */
    Ptr<OddLabelMap> m_routingLabelMap;
};

} // namespace ns3

#endif  // NS3_PACKERL_H