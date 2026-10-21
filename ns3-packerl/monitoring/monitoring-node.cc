#include "monitoring-node.h"

using std::make_pair;

namespace ns3
{
NS_LOG_COMPONENT_DEFINE("MonitoringNode");

MonitoringNode::MonitoringNode(uint32_t nodeId)
    : m_nodeId(nodeId)
{
    NS_LOG_FUNCTION(this);
    resetStats();
}

void
MonitoringNode::resetStats()
{
    NS_LOG_FUNCTION(this);

    m_txPerReceiver.clear();
    m_reTxPerReceiver.clear();
    m_rxPerSender.clear();
    m_deliveredPerSender.clear();
    m_ttlDroppedPerPair.clear();
    m_tcpDiscardedPerSender.clear();
}

void
MonitoringNode::registerTcpRetransmit(uint32_t dstNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << dstNodeId << bytes);
    m_reTxPerReceiver[dstNodeId] += bytes;
    m_cumulReTxPerReceiver[dstNodeId] += bytes;
}

void
MonitoringNode::registerUdpRx(uint32_t srcNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << srcNodeId << bytes);
    m_deliveredPerSender[srcNodeId] += bytes;
    m_cumulDeliveredPerSender[srcNodeId] += bytes;
}

void
MonitoringNode::registerTcpRx(uint32_t srcNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << srcNodeId << bytes);
    m_deliveredPerSender[srcNodeId] += bytes;
    m_cumulDeliveredPerSender[srcNodeId] += bytes;
}

void
MonitoringNode::registerTcpRxDiscard(uint32_t srcNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << srcNodeId << bytes);
    m_tcpDiscardedPerSender[srcNodeId] += bytes;
    m_cumulTcpDiscardedPerSender[srcNodeId] += bytes;
}

void
MonitoringNode::registerFlowmonSent(uint32_t dstNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << dstNodeId << bytes);
    m_txPerReceiver[dstNodeId] += bytes;
    m_cumulTxPerReceiver[dstNodeId] += bytes;
}

void
MonitoringNode::registerFlowmonReceived(uint32_t srcNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << srcNodeId << bytes);
    m_rxPerSender[srcNodeId] += bytes;
    m_cumulRxPerSender[srcNodeId] += bytes;
}

void
MonitoringNode::registerFlowmonDrop(uint32_t srcNodeId, uint32_t dstNodeId, uint32_t bytes)
{
    NS_LOG_FUNCTION(this << srcNodeId << dstNodeId << bytes);
    m_ttlDroppedPerPair[make_pair(srcNodeId, dstNodeId)] += bytes;
    m_cumulTtlDroppedPerPair[make_pair(srcNodeId, dstNodeId)] += bytes;
}

}