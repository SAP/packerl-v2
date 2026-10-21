#include "monitoring-edge.h"

#include "ns3/ipv4-flow-probe-tag.h"


namespace ns3
{
NS_LOG_COMPONENT_DEFINE("MonitoringEdge");

MonitoringEdge::MonitoringEdge(U32Pair src,
                               U32Pair dst,
                               uint32_t channelDataRate,
                               uint32_t channelDelay,
                               uint32_t txQueueCapacity)
    : m_src(src),
      m_dst(dst),
      m_channelDataRate(channelDataRate),
      m_channelDelay(channelDelay),
      m_txQueueCapacity(txQueueCapacity),
      m_isLinkUp(true)
{
    resetStats();
}

void
MonitoringEdge::resetStats()
{
    NS_LOG_FUNCTION(this);
    m_transmitted = 0;
    m_macTxDropped = 0;
    m_phyRxDropped = 0;
}

void
MonitoringEdge::registerLinkDown()
{
    NS_LOG_FUNCTION(this);
    m_isLinkUp = false;
}

void
MonitoringEdge::registerTransmission(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    Ipv4FlowProbeTag fTag;
    if (! p->FindFirstMatchingByteTag(fTag))
    {
        NS_LOG_LOGIC("MonitoringEdge::registerMacTxDrop(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    m_transmitted += p->GetSize();
}

void
MonitoringEdge::registerMacTxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    Ipv4FlowProbeTag fTag;
    if (! p->FindFirstMatchingByteTag(fTag))
    {
        NS_LOG_LOGIC("MonitoringEdge::registerMacTxDrop(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    m_macTxDropped += fTag.GetAppDataSize();
}

void
MonitoringEdge::registerPhyTxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    NS_FATAL_ERROR("registerPhyTxDrop(): Not currently supported. These shouldn't happen.");
}

void
MonitoringEdge::registerPhyRxDrop(Ptr<const Packet> p)
{
    NS_LOG_FUNCTION(this << p->GetUid());
    Ipv4FlowProbeTag fTag;
    if (! p->FindFirstMatchingByteTag(fTag))
    {
        NS_LOG_LOGIC("MonitoringEdge::registerMacTxDrop(): packet has no Ipv4FlowProbeTag. This may be a meta-packet.");
        return;
    }
    m_phyRxDropped += fTag.GetAppDataSize();
}

}