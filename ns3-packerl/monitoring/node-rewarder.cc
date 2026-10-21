#include "node-rewarder.h"


namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("NodeRewarder");
NS_OBJECT_ENSURE_REGISTERED (NodeRewarder);

TypeId
NodeRewarder::GetTypeId ()
{
    static TypeId tid = TypeId("ns3::NodeRewarder")
                            .SetParent<Object>()
                            .AddConstructor<NodeRewarder>();
    return tid;
}

NodeRewarder::NodeRewarder()
{
    NS_LOG_FUNCTION(this);
}

void
NodeRewarder::registerNode(uint32_t nodeId)
{
    NS_LOG_FUNCTION(this);
    if (nodeId != m_nodeRewards.size())
    {
        NS_FATAL_ERROR("NodeRewarder: Node IDs must be continuous and start at 0");
    }
    m_nodeRewards[nodeId] = 0.0;
}

void
NodeRewarder::distributeRewards(double baseReward, const PacketPath& path, bool excludeLastNode)
{
    NS_LOG_FUNCTION(this << baseReward << "PATH" << path);

    // Early exit for trivial cases
    if (path.empty() || baseReward == 0.0 || (excludeLastNode && path.size() == 1)) return;

    // Create a view of the relevant nodes (if desired, prune the path if it's too long)
    auto pathBegin = path.begin();
    auto pathEnd = excludeLastNode ? path.end() - 1 : path.end();
    if (m_pathCutoff > 0 && std::distance(pathBegin, pathEnd) > m_pathCutoff)
    {
        pathBegin = pathEnd - m_pathCutoff;
    }

    // if capping the reward sum to bytes, calculate starting reward depending on path length.
    // this ensures that, if m_capToBytes is set, the sum of rewards for all nodes in the path is equal to baseReward.
    double relativeSum = 0.0;
    double relativeInc = 1.0;
    for (auto it = pathBegin; it != pathEnd; ++it)
    {
        relativeSum += relativeInc;
        relativeInc *= m_spatialDecay;
    }
    double reward = m_capToBytes ? (baseReward / relativeSum) : baseReward;

    // distribute rewards along the path
    double rewardDistributed = 0.0;
    for (auto it = std::make_reverse_iterator(pathEnd); it != std::make_reverse_iterator(pathBegin); ++it)
    {
        m_nodeRewards[*it] += reward;
        rewardDistributed += reward;
        reward *= m_spatialDecay;
    }
}

void
NodeRewarder::HandleAppRx(uint32_t payloadSize, const PacketPath& path)
{
    NS_LOG_FUNCTION(this << payloadSize << path);
    distributeRewards(static_cast<double>(payloadSize) / 1e6, path);
}

void
NodeRewarder::HandleTcpRxDiscard(uint32_t payloadSize, const PacketPath& path)
{
    NS_LOG_FUNCTION(this << payloadSize << path);
    distributeRewards(-static_cast<double>(payloadSize) / 1e6, path);
}

void
NodeRewarder::HandleDropTTL(uint32_t payloadSize, const PacketPath& path)
{
    NS_LOG_FUNCTION(this << payloadSize << path);
    distributeRewards(-static_cast<double>(payloadSize) / 1e6, path);
}

void
NodeRewarder::HandleDropQueue(uint32_t payloadSize, const PacketPath& path, bool wasUnavoidable)
{
    NS_LOG_FUNCTION(this << payloadSize << path);
    // don't penalize the queueing node if the drop was unavoidable (wasUnavoidable = excludeLastNode)
    distributeRewards(-static_cast<double>(payloadSize) / 1e6, path, wasUnavoidable);
}

void
NodeRewarder::HandleDropPhyRx(uint32_t payloadSize, const PacketPath& path)
{
    NS_LOG_FUNCTION(this << payloadSize << path);
    distributeRewards(-static_cast<double>(payloadSize) / 1e6, path);
}

void
NodeRewarder::ResetRewards()
{
    NS_LOG_FUNCTION(this);
    for (auto &nodeReward : m_nodeRewards)
    {
        nodeReward.second = 0.0;
    }
}

}   // namespace ns3