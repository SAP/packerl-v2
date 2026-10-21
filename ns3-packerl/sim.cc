#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/ns3-ai-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/point-to-point-module.h"

#include "packerl.h"

#include <iostream>
#include <optional>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("PackerlSim");

/**
 * Main method of the PackeRL sim.
 * Gets called at the start of every episode, and after termination the episode ends.
 */
int
main(int argc, char* argv[])
{
    uint32_t run = 0;
    uint32_t memblockKey = 0;
    uint32_t mtu = 1500;
    uint32_t ipv4HeaderSize = 20;
    uint32_t udpHeaderSize = 8;
    uint32_t tcpHeaderSize = 40;
    int birdseyeObserverId = -1;
    double localRewardSpatialDecay = 1.0;
    uint32_t localRewardPathCutoff = 0;
    bool localRewardCapToBytes = false;
    double simStepDuration = 0.0;
    double bufferSizeBdp = 0.0;
    double probTcp = 0.0;
    bool useTcpSack = false;
    bool capLinkUtilization = false;
    bool probabilisticRouting = false; // (currently not used)

    CommandLine cmd;
    cmd.AddValue("run", "Run index (for setting repeatable seeds)", run);
    cmd.AddValue("memblockKey", "Memblock key for interaction with ns3-ai", memblockKey);
    cmd.AddValue("mtu", "Maximum Transmission Unit (MTU) of Ethernet in bytes", mtu);
    cmd.AddValue("ipv4HeaderSize", "IPv4 header size in bytes", ipv4HeaderSize);
    cmd.AddValue("udpHeaderSize", "UDP header size in bytes", udpHeaderSize);
    cmd.AddValue("tcpHeaderSize", "TCP header size in bytes", tcpHeaderSize);
    cmd.AddValue("birdseyeObserverId", "Observer ID for birdseye view", birdseyeObserverId);
    cmd.AddValue("localRewardSpatialDecay", "Spatial decay factor for local reward", localRewardSpatialDecay);
    cmd.AddValue("localRewardPathCutoff", "Path length cutoff for local reward", localRewardPathCutoff);
    cmd.AddValue("localRewardCapToBytes", "Cap local reward to bytes", localRewardCapToBytes);
    cmd.AddValue("simStepDuration", "Simulation step duration in ms", simStepDuration);
    cmd.AddValue("bufferSizeBdp", "Buffer size in multiples of Bandwidth-RTT-product (bytes)", bufferSizeBdp);
    cmd.AddValue("probTcp", "Probability of using TCP (0.0 to 1.0)", probTcp);
    cmd.AddValue("useTcpSack", "Use TCP SACK (true/false)", useTcpSack);
    cmd.AddValue("capLinkUtilization", "Cap link utilization (true/false)", capLinkUtilization);
    cmd.AddValue("probabilisticRouting", "Use probabilistic routing (true/false)", probabilisticRouting);

    cmd.Parse(argc, argv);

    SimParameters simParameters;
    simParameters.memblockKey = memblockKey;
    simParameters.mtu = mtu;
    simParameters.ipv4HeaderSize = ipv4HeaderSize;
    simParameters.udpHeaderSize = udpHeaderSize;
    simParameters.tcpHeaderSize = tcpHeaderSize;
    simParameters.birdseyeObserverId = birdseyeObserverId;
    simParameters.localRewardSpatialDecay = localRewardSpatialDecay;
    simParameters.localRewardPathCutoff = localRewardPathCutoff;
    simParameters.localRewardCapToBytes = localRewardCapToBytes;
    simParameters.simStepDuration = simStepDuration;
    simParameters.bufferSizeBdp = bufferSizeBdp;
    simParameters.probTcp = probTcp;
    simParameters.useTcpSack = useTcpSack;
    simParameters.capLinkUtilization = capLinkUtilization;
    simParameters.probabilisticRouting = probabilisticRouting;

    NS_LOG_INFO("shm sizes: act " << sizeof(PackerlActStruct) << ", env " << sizeof(PackerlEnvStruct) );

    RngSeedManager::SetRun(run);

    // set up env, and signal readiness to python side
    PackerlEnv env(simParameters);
    env.EnvSetterCond()->simReady = true;
    env.SetCompleted();
    NS_LOG_INFO("initialized PackeRL");

    // run the simulator loop
    bool ok = env.runSimLoop();

    // experiment cleanup
    NS_LOG_INFO("Exited simLoop (ok=" << ok << "), destroying simulation...");
    Simulator::Destroy();
    return ok ? 0 : 1;
}
