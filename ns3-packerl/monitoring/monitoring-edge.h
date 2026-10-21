#ifndef NS3_MONITORING_EDGE_H
#define NS3_MONITORING_EDGE_H

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "../utils/types.h"
#include "../utils/shared-structs.h"

using std::make_pair;

namespace ns3
{

/**
* A directed edge that monitors the network performance along a channel
 * spanned between two given nodes. Implements trace sinks that register sent,
 * received and dropped packets at the lower levels. The edge is directed because
 * we model only one sending direction; thus, for a channel with bidirectional traffic,
 * we need two such MonitoringEdges in both orientations.
 */
class MonitoringEdge : public SimpleRefCount<MonitoringEdge>
{
  public:
    /**
     * Constructor taking basic channel/edge configuration
     * @param src source IDs (node and device)
     * @param dst destination IDs (node and device)
     * @param channelDataRate channel data rate
     * @param channelDelay channel delay
     * @param txQueueCapacity capacity of transmission NetDevice queue in bytes
     */
    MonitoringEdge(U32Pair src,
                   U32Pair dst,
                   uint32_t channelDataRate,
                   uint32_t channelDelay,
                   uint32_t txQueueCapacity);

    /**
     * Resets the monitored statistics to default values
     */
    void resetStats();

    /**
     * Trace sink for link down events
     */
    void registerLinkDown();

    /**
     * Trace sink for successful packet transmissions from source NetDevice.
     * @param p sent packet
     */
    void registerTransmission(Ptr<const Packet> p);

    /**
     * Trace sink for dropped packet before transmission (e.g. TxQueue enqueueing)
     * @param p dropped packet
     */
    void registerMacTxDrop(Ptr<const Packet> p);

    /**
     * Trace sink for dropped packet during transmission
     * @param p dropped packet
     */
    void registerPhyTxDrop(Ptr<const Packet> p);

    /**
     * Trace sink for dropped packet during receival
     * @param p dropped packet
     */
    void registerPhyRxDrop(Ptr<const Packet> p);

    /**
     * Getters for monitored values
     */
    U32Pair getSrc() const { return m_src; }
    U32Pair getDst() const { return m_dst; }
    U32Pair getSrcDstNodes() const { return make_pair(m_src.first, m_dst.first); }

    uint32_t getChannelDataRate() const { return m_channelDataRate; }
    uint32_t getChannelDelay() const { return m_channelDelay; }
    uint32_t getTxQueueCapacity() const { return m_txQueueCapacity; }

    uint64_t getTransmitted() const { return m_transmitted; }
    uint64_t getMacTxDropped() const { return m_macTxDropped; }
    uint64_t getPhyRxDropped() const { return m_phyRxDropped; }

    bool isLinkUp() const { return m_isLinkUp; }

  private:

    // ===== monitored values =====

    // --- Configuration ---

    U32Pair m_src;  // source node and device ID
    U32Pair m_dst;  // destination node and device ID
    uint32_t m_channelDataRate;  // channel data rate in bps
    uint32_t m_channelDelay;  // channel delay in ms
    uint32_t m_txQueueCapacity;  // source NetDevice's transmission queue capacity in bytes

    // --- Statistics ---

    uint64_t m_transmitted;   // total bytes transmitted over the channel (includes all headers and overhead)
    uint64_t m_macTxDropped;  // dropped at MAC layer (transmission)
    uint64_t m_phyRxDropped;  // dropped at PHY layer (reception)

    bool m_isLinkUp;  // whether the link is up or down
};

}

#endif // NS3_MONITORING_EDGE_H