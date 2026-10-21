#ifndef NS3_PACKERL_NODE_REWARDER_H
#define NS3_PACKERL_NODE_REWARDER_H

#include "ns3/core-module.h"

#include <unordered_map>
#include <vector>

using std::unordered_map, std::vector, std::pair;

typedef vector<uint32_t> PacketPath;

namespace ns3
{

/**
 * @ingroup packerl
 * @brief NodeRewarder is responsible for distributing rewards to nodes based on their participation in packet paths.
 *
 * The NodeRewarder distributes rewards to nodes that are part of a packet's path, applying spatial decay
 * and optionally capping rewards to the size of the payload.
 */
class NodeRewarder : public Object
{
    public:
        static TypeId GetTypeId(void);
        NodeRewarder();

        void SetSpatialDecay(double spatialDecay) { m_spatialDecay = spatialDecay; };
        void SetPathCutoff(uint32_t pathCutoff) { m_pathCutoff = pathCutoff; };
        void SetCapToBytes(bool capToBytes) { m_capToBytes = capToBytes; };

        /**
         * Registers a node with the NodeRewarder.
         * @param nodeId The ID of the node to register.
         * This method should be called for each node that will be part of the reward distribution.
         */
        void registerNode(uint32_t nodeId);

        /**
         * Spatially distributes rewards to nodes along the packet's path.
         * @param baseReward The base reward to distribute.
         * @param path The path of nodes to distribute the reward to.
         * @param excludeLastNode If true, the last node (the event node) in the path will not receive a reward.
         */
        void distributeRewards(double baseReward, const PacketPath& path, bool excludeLastNode = false);

        void HandleAppRx(uint32_t payloadSize, const PacketPath& path);
        void HandleTcpRxDiscard(uint32_t payloadSize, const PacketPath& path);
        void HandleDropTTL(uint32_t payloadSize, const PacketPath& path);
        void HandleDropQueue(uint32_t payloadSize, const PacketPath& path, bool wasUnavoidable);
        void HandleDropPhyRx(uint32_t payloadSize, const PacketPath& path);

        // Resets all node rewards to zero
        void ResetRewards();

        /**
         * Returns the node rewards
         */
        const std::unordered_map<uint32_t, double>& GetNodeRewards() const { return m_nodeRewards; };

    private:
        /**
         * Timestamp of the last monitoring in seconds
         */
        std::unordered_map<uint32_t, double> m_nodeRewards;

        double m_spatialDecay = 1.0;
        uint32_t m_pathCutoff = 0;
        bool m_capToBytes = false;
};

}  // namespace ns3

#endif // NS3_PACKERL_NODE_REWARDER_H
