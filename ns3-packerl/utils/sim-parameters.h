#ifndef NS3_PACKERL_SIM_PARAMETERS_H
#define NS3_PACKERL_SIM_PARAMETERS_H


/**
 * A custom struct holding ns3 simulation parameters,
 * used within the PackeRL framework for repeatable RL experiments.
 */
typedef struct SimParameters {
    /**
     * Memory block key, shared with the python side of the framework
     */
    uint32_t memblockKey;

    /**
     * Simulation time per step in ms
     */
    double simStepDuration;

    /**
     * MTU size in bytes
     */
    uint32_t mtu;

    /**
     * Ipv4 header size in bytes
     */
    uint32_t ipv4HeaderSize;

    /**
     * UDP header size in bytes
     */
    uint32_t udpHeaderSize;

    /**
     * TCP header size in bytes
     */
    uint32_t tcpHeaderSize;

    /**
     * The ID of the birdseye observer.
     */
    int birdseyeObserverId;

    /**
     * Spatial reward decay factor; reward gets multiplied by this factor for each hop away from the source.
     */
    double localRewardSpatialDecay;

    /**
     * Max path length for local reward propagation
     */
    uint32_t localRewardPathCutoff;

    /**
     * If true, the sum of local rewards assigned to all nodes from a single packet is capped to that packet's size.
     */
    bool localRewardCapToBytes;

    /**
     * Buffer size in BDPs
     */
    double bufferSizeBdp;

    /**
     * Whether to enable TCP SACKs or not
     */
    bool useTcpSack;

    /**
     * Link utilization per step is calculated via
     * <bytes transferred> / <bytes that could have been transferred in this step>.
     * Since the latter value is the product of link datarate and elapsed time,
     * the entire result can exceed 1.0 if packets arrive very early in the time step.
     * If this is set to true, the link utilization is capped at 1.0.
     */
    bool capLinkUtilization;

    /**
     * The probability of a flow being TCP (instead of UDP).
     */
    double probTcp;

    /**
     * Whether to use probabilistic routing or not
     */
    bool probabilisticRouting;
} SimParameters;

#endif // NS3_PACKERL_SIM_PARAMETERS_H
