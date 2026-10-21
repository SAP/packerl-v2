#include "global-rewarder.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("GlobalRewarder");
NS_OBJECT_ENSURE_REGISTERED(GlobalRewarder);

TypeId
GlobalRewarder::GetTypeId()
{
    static TypeId tid = TypeId("GlobalRewarder")
                            .SetParent<Object>()
                            .AddConstructor<GlobalRewarder>();
    return tid;
}

GlobalRewarder::GlobalRewarder()
{
    NS_LOG_FUNCTION(this);
}

void
GlobalRewarder::SetWeights(const GlobalRewardWeights& weights)
{
    NS_LOG_FUNCTION(this);
    m_weights = weights;

    NS_LOG_DEBUG("+ --- GlobalRewardWeights -------");
    NS_LOG_DEBUG("| tx: " << m_weights.txWeight);
    NS_LOG_DEBUG("| reTx: " << m_weights.reTxWeight);
    NS_LOG_DEBUG("| inFlight: " << m_weights.inFlightWeight);
    NS_LOG_DEBUG("| rx: " << m_weights.rxWeight);
    NS_LOG_DEBUG("| delivered: " << m_weights.deliveredWeight);
    NS_LOG_DEBUG("| ttlDropped: " << m_weights.ttlDroppedWeight);
    NS_LOG_DEBUG("| tcpDiscarded: " << m_weights.tcpDiscardedWeight);
    NS_LOG_DEBUG("| givenUp: " << m_weights.givenUpWeight);
    NS_LOG_DEBUG("| ");
    NS_LOG_DEBUG("| maxDelay: " << m_weights.maxDelayWeight);
    NS_LOG_DEBUG("| avgDelay: " << m_weights.avgDelayWeight);
    NS_LOG_DEBUG("| avgJitter: " << m_weights.avgJitterWeight);
    NS_LOG_DEBUG("| avgTcpSenderSrtt: " << m_weights.avgTcpSenderSrttWeight);
    NS_LOG_DEBUG("| ");
    NS_LOG_DEBUG("| dropRatio: " << m_weights.dropRatioWeight);
    NS_LOG_DEBUG("| inFlightRatio: " << m_weights.inFlightRatioWeight);
    NS_LOG_DEBUG("| maxLinkUtilization: " << m_weights.maxLinkUtilizationWeight);
    NS_LOG_DEBUG("| avgLinkUtilization: " << m_weights.avgLinkUtilizationWeight);
    NS_LOG_DEBUG("| maxTxQueueRelativeLoad: " << m_weights.maxTxQueueRelativeLoadWeight);
    NS_LOG_DEBUG("| avgTxQueueRelativeLoad: " << m_weights.avgTxQueueRelativeLoadWeight);
    NS_LOG_DEBUG("| ");
    NS_LOG_DEBUG("| transmitted: " << m_weights.transmittedWeight);
    NS_LOG_DEBUG("| macTxDropped: " << m_weights.macTxDroppedWeight);
    NS_LOG_DEBUG("| phyRxDropped: " << m_weights.phyRxDroppedWeight);
    NS_LOG_DEBUG("+ -------------------------------");
}

double
GlobalRewarder::CalculateReward(ObservedGlobal& observedGlobal)
{
    NS_LOG_FUNCTION(this);
    double reward = 0.0;

    // we calculate the reward components of byte values in MB

    reward += m_weights.txWeight * static_cast<double>(observedGlobal.tx) / 1e6;
    reward += m_weights.reTxWeight * static_cast<double>(observedGlobal.reTx) / 1e6;
    reward += m_weights.inFlightWeight * static_cast<double>(observedGlobal.inFlight) / 1e6;
    reward += m_weights.rxWeight * static_cast<double>(observedGlobal.rx) / 1e6;
    reward += m_weights.deliveredWeight * static_cast<double>(observedGlobal.delivered) / 1e6;
    reward += m_weights.ttlDroppedWeight * static_cast<double>(observedGlobal.ttlDropped) / 1e6;
    reward += m_weights.tcpDiscardedWeight * static_cast<double>(observedGlobal.tcpDiscarded) / 1e6;
    reward += m_weights.givenUpWeight * static_cast<double>(observedGlobal.givenUp) / 1e6;

    reward += m_weights.maxDelayWeight * static_cast<double>(observedGlobal.maxDelay);
    reward += m_weights.avgDelayWeight * static_cast<double>(observedGlobal.avgDelay);
    reward += m_weights.avgJitterWeight * static_cast<double>(observedGlobal.avgJitter);
    reward += m_weights.avgTcpSenderSrttWeight * static_cast<double>(observedGlobal.avgTcpSenderSrtt);

    reward += m_weights.dropRatioWeight * static_cast<double>(observedGlobal.dropRatio);
    reward += m_weights.inFlightRatioWeight * static_cast<double>(observedGlobal.inFlightRatio);
    reward += m_weights.maxLinkUtilizationWeight * static_cast<double>(observedGlobal.maxLinkUtilization);
    reward += m_weights.avgLinkUtilizationWeight * static_cast<double>(observedGlobal.avgLinkUtilization);
    reward += m_weights.maxTxQueueRelativeLoadWeight * static_cast<double>(observedGlobal.maxTxQueueRelativeLoad);
    reward += m_weights.avgTxQueueRelativeLoadWeight * static_cast<double>(observedGlobal.avgTxQueueRelativeLoad);

    reward += m_weights.transmittedWeight * static_cast<double>(observedGlobal.transmitted) / 1e6;
    reward += m_weights.macTxDroppedWeight * static_cast<double>(observedGlobal.macTxDropped) / 1e6;
    reward += m_weights.phyRxDroppedWeight * static_cast<double>(observedGlobal.phyRxDropped) / 1e6;

    return reward;
}

}  // namespace ns3