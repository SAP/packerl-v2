#ifndef NS3_MONITORING_NODE_H
#define NS3_MONITORING_NODE_H

#include <stdint.h>
#include <vector>

#include "ns3/core-module.h"
#include "ns3/internet-module.h"

#include "../utils/types.h"
#include "../utils/shared-structs.h"

using namespace ns3;

namespace ns3
{

/**
 * A class to monitor statistics of a node in the network. It tracks various
 * lower-level trace events happening at the corresponding simulation node.
 */
class MonitoringNode : public SimpleRefCount<MonitoringNode>
{
  public:

    /**
     * Constructor
     * @param nodeId The node ID
     */
    MonitoringNode(uint32_t nodeId);

    /**
     * Resets the monitored statistics to default values
     */
    void resetStats();

    /**
     * @return The node ID
     */
    uint32_t GetNodeId() const { return m_nodeId; };

    const U32ToU64Map& GetTxPerReceiver() const { return m_txPerReceiver; };
    const U32ToU64Map& GetReTxPerReceiver() const { return m_reTxPerReceiver; };
    const U32ToU64Map& GetRxPerSender() const { return m_rxPerSender; };
    const U32ToU64Map& GetDeliveredPerSender() const { return m_deliveredPerSender; };
    const U32PairToU64Map& GetTtlDroppedPerPair() const { return m_ttlDroppedPerPair; };
    const U32ToU64Map& GetTcpDiscardedPerSender() const { return m_tcpDiscardedPerSender; };

    const U32ToU64Map& GetCumulTxPerReceiver() const { return m_cumulTxPerReceiver; };
    const U32ToU64Map& GetCumulReTxPerReceiver() const { return m_cumulReTxPerReceiver; };
    const U32ToU64Map& GetCumulRxPerSender() const { return m_cumulRxPerSender; };
    const U32ToU64Map& GetCumulDeliveredPerSender() const { return m_cumulDeliveredPerSender; };
    const U32PairToU64Map& GetCumulTtlDroppedPerPair() const { return m_cumulTtlDroppedPerPair; };
    const U32ToU64Map& GetCumulTcpDiscardedPerSender() const { return m_cumulTcpDiscardedPerSender; };

    void registerTcpRetransmit(uint32_t dstNodeId, uint32_t bytes);
    void registerUdpRx(uint32_t srcNodeId, uint32_t bytes);
    void registerTcpRx(uint32_t srcNodeId, uint32_t bytes);
    void registerTcpRxDiscard(uint32_t srcNodeId, uint32_t bytes);
    void registerFlowmonSent(uint32_t dstNodeId, uint32_t bytes);
    void registerFlowmonReceived(uint32_t srcNodeId, uint32_t bytes);
    void registerFlowmonDrop(uint32_t srcNodeId, uint32_t dstNodeId, uint32_t bytes);

  private:

    uint32_t m_nodeId;

    U32ToU64Map m_txPerReceiver;  // as sender node
    U32ToU64Map m_reTxPerReceiver;  // as sender node
    U32ToU64Map m_rxPerSender;  // as receiver node
    U32ToU64Map m_deliveredPerSender;  // as receiver node
    U32PairToU64Map m_ttlDroppedPerPair;  // as dropping node
    U32ToU64Map m_tcpDiscardedPerSender;  // as receiver node

    U32ToU64Map m_cumulTxPerReceiver;  // as sender node
    U32ToU64Map m_cumulReTxPerReceiver;  // as sender node
    U32ToU64Map m_cumulRxPerSender;  // as receiver node
    U32ToU64Map m_cumulDeliveredPerSender;  // as receiver node
    U32PairToU64Map m_cumulTtlDroppedPerPair;  // as dropping node
    U32ToU64Map m_cumulTcpDiscardedPerSender;  // as receiver node
};

}

#endif // NS3_MONITORING_NODE_H