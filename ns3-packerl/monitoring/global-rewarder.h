#ifndef NS3_PACKERL_GLOBAL_REWARDER_H
#define NS3_PACKERL_GLOBAL_REWARDER_H

#include "ns3/core-module.h"

#include "../utils/shared-structs.h"

namespace ns3
{

/**
 * \ingroup packerl
 * \brief Class to calculate global rewards based on observed global state.
 *
 * This class uses a set of weights to compute a reward value from an observed global state.
 */
class GlobalRewarder : public Object
{
    public:
        static TypeId GetTypeId(void);
        GlobalRewarder();

        void SetWeights(const GlobalRewardWeights& weights);

        double CalculateReward(ObservedGlobal& observedGlobal);

    private:

    GlobalRewardWeights m_weights;

};

}  // namespace ns3

#endif  // NS3_PACKERL_GLOBAL_REWARDER_H