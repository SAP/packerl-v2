"""
Features are what we measure from the ns-3 simulator and our own actions.

A. They come in different value ranges (first entry of the value tuple, used for visualization):
 "unit" values range from 0 to 1,
 "small" values are not bounded but typically small (abs < 1),
 "large" values are not bounded and typically large (abs > 1).

B. The second entry of the tuple is the type of the feature. It can be one of the following:
    "s" for spec,
    "b" for bytes,
    "c" for channel,
    "d" for delay,
    "u" for upcoming flow arrivals,
    "l" for load,
    "r" for ratio,
    "a" for auxiliary.
    
C. For node features, the third entry of the tuple indicates whether the feature is ego-centric or not.

For obtaining the features that will be provided to the policy, we import the corresponding struct dtypes to ensure
That the feature order is consistent with the order of the features in the struct.
"""
from omegaconf import DictConfig

from src.utils.shared_memory.structs import observed_global_dtype, observed_node_dtype, observed_edge_dtype


ALL_GLOBAL_FEATURES = {

    # bytes  [b]
    "tx":                               ("large", 'b'),  # app data bytes sent out for routing [throughput]
    "reTx":                             ("large", 'b'),  # app data bytes retransmitted (e.g. TCP)
    "inFlight":                         ("large", 'b'),  # in-flight app data bytes
    "rx":                               ("large", 'b'),  # app data bytes received (but not necessarily delivered to the sink app)
    "delivered":                        ("large", 'b'),  # app data bytes delivered to the sink app
    "ttlDropped":                       ("large", 'b'),  # app data bytes dropped due to TTL exceeded
    "tcpDiscarded":                     ("large", 'b'),  # app data bytes received but discarded by TCP (e.g. due to out-of-order)
    "givenUp":                          ("large", 'b'),  # app data bytes dropped (all reasons, incl. TCP discards)

    # delay  [d]
    "maxDelay":                         ("small", 'd'),  # maximum delay of received packets
    "avgDelay":                         ("small", 'd'),  # average delay of received packets
    "avgJitter":                        ("small", 'd'),  # average jitter of received packets
    "avgTcpSenderSrtt":                 ("small", 'd'),  # average TCP sender smoothed RTT

     # upcoming flows  [u]
    "udpRemainingFlows":                ("large", 'u'),  # number of active UDP flows
    "udpRemainingFlowBytes":            ("large", 'u'),  # remaining bytes of active UDP flows
    "udpRemainingFlowDatarateSum":      ("large", 'u'),  # current cumulative sending rate of all UDP flows
    "tcpRemainingFlows":                ("large", 'u'),  # number of active TCP flows
    "tcpRemainingFlowBytes":            ("large", 'u'),  # remaining bytes of active TCP flows
    "tcpRemainingFlowCWndSum":          ("large", 'u'),  # current cumulative congestion window of all TCP flows

    # channel  [c]
    "transmitted":                      ("large", 'c'),  # gross bytes transmitted by channels (incl. all headers and overhead)
    "transmittable":                    ("large", 'c'),  # amount of bytes that can be transmitted via channels in the current time frame
    "macTxDropped":                     ("large", 'c'),  # app data bytes dropped during enqueueing in network devices
    "phyRxDropped":                     ("large", 'c'),  # app data bytes dropped during reception in network devices

    # ratio  [r]
    "dropRatio":                        ( "unit", 'r'),  # ratio of given-up packets to given-up + delivered packets
    "inFlightRatio":                    ( "unit", 'r'),  # ratio of in-flight packets to throughput

    # load  [l]
    "maxLinkUtilization":               ( "unit", 'l'),  # maximum link utilization
    "avgLinkUtilization":               ( "unit", 'l'),  # average link utilization
    "maxTxQueueRelativeLoad":           ( "unit", 'l'),  # maximum relative load of all transmission queues
    "avgTxQueueRelativeLoad":           ( "unit", 'l'),  # average relative load of all transmission queues

    # auxiliary  [a]
    "delayCount":                       ("large", 'a'),  # number of delay values
    "jitterCount":                      ("large", 'a'),  # number of jitter values
    "tcpSenderSrttCount":               ("large", 'a'),  # number of TCP sender smoothed RTT values
    "txQueueCapacity":                  ("large", 'a'),  # capacity of transmission queues
    "txQueueLoad":                      ("large", 'a'),  # load of transmission queues
}

global_feat_names = list(ALL_GLOBAL_FEATURES.keys())
global_feat_in_struct_order = [feat for feat in observed_global_dtype.names if feat in global_feat_names]


ALL_NODE_FEATURES = {

    # === ego-perspective features ===

    # bytes  [b]
    "txAsSender":                               ("large", 'b', True),
    "reTxAsSender":                             ("large", 'b', True),
    "rxAsReceiver":                             ("large", 'b', True),
    "deliveredAsReceiver":                      ("large", 'b', True),
    "ttlDroppedAsDropper":                      ("large", 'b', True),
    "tcpDiscardedAsReceiver":                   ("large", 'b', True),
    "givenUpAsDropper":                         ("large", 'b', True),

    # delay  [d]
    "maxDelayAsReceiver":                       ("small", 'd', True),
    "avgDelayAsReceiver":                       ("small", 'd', True),
    "avgJitterAsReceiver":                      ("small", 'd', True),
    "avgTcpSenderSrttAsSender":                 ("small", 'd', True),

    # upcoming flows  [u]
    "udpRemainingFlowsAsSender":                ("large", 'u', True),
    "udpRemainingFlowBytesAsSender":            ("large", 'u', True),
    "udpRemainingFlowDatarateSumAsSender":      ("large", 'u', True),
    "tcpRemainingFlowsAsSender":                ("large", 'u', True),
    "tcpRemainingFlowBytesAsSender":            ("large", 'u', True),
    "tcpRemainingFlowCWndSumAsSender":          ("large", 'u', True),

    # channel  [c]
    "transmittedAsEdgeSrc":                     ("large", 'c', True),
    "transmittableAsEdgeSrc":                   ("large", 'c', True),
    "macTxDroppedAsEdgeSrc":                    ("large", 'c', True),
    "phyRxDroppedAsEdgeDst":                    ("large", 'c', True),

    # load  [l]
    "maxLinkUtilizationAsEdgeSrc":              ( "unit", 'l', True),
    "avgLinkUtilizationAsEdgeSrc":              ( "unit", 'l', True),
    "maxTxQueueRelativeLoadAsEdgeSrc":          ( "unit", 'l', True),
    "avgTxQueueRelativeLoadAsEdgeSrc":          ( "unit", 'l', True),

    # === features requiring other perspectives ===

    # bytes  [b]
    "txAsReceiver":                             ("large", 'b', False),
    "reTxAsReceiver":                           ("large", 'b', False),
    "inFlightAsSender":                         ("large", 'b', False),
    "inFlightAsReceiver":                       ("large", 'b', False),
    "rxAsSender":                               ("large", 'b', False),
    "deliveredAsSender":                        ("large", 'b', False),
    "ttlDroppedAsSender":                       ("large", 'b', False),
    "ttlDroppedAsReceiver":                     ("large", 'b', False),
    "tcpDiscardedAsSender":                     ("large", 'b', False),
    "givenUpAsSender":                          ("large", 'b', False),
    "givenUpAsReceiver":                        ("large", 'b', False),

    # delay  [d]
    "maxDelayAsSender":                         ("small", 'd', False),
    "avgDelayAsSender":                         ("small", 'd', False),
    "avgJitterAsSender":                        ("small", 'd', False),
    "avgTcpSenderSrttAsReceiver":               ("small", 'd', False),

    # upcoming flows  [u]
    "udpRemainingFlowsAsReceiver":              ("large", 'u', False),
    "udpRemainingFlowBytesAsReceiver":          ("large", 'u', False),
    "udpRemainingFlowDatarateSumAsReceiver":    ("large", 'u', False),
    "tcpRemainingFlowsAsReceiver":              ("large", 'u', False),
    "tcpRemainingFlowBytesAsReceiver":          ("large", 'u', False),
    "tcpRemainingFlowCWndSumAsReceiver":        ("large", 'u', False),

    # channel  [c]
    "transmittedAsEdgeDst":                     ("large", 'c', False),
    "transmittableAsEdgeDst":                   ("large", 'c', False),
    "macTxDroppedAsEdgeDst":                    ("large", 'c', False),
    "phyRxDroppedAsEdgeSrc":                    ("large", 'c', False),

    # ratio [r]
    "dropRatioAsSender":                        ("unit", 'r', False),
    "dropRatioAsReceiver":                      ("unit", 'r', False),
    "inFlightRatioAsSender":                    ("unit", 'r', False),
    "inFlightRatioAsReceiver":                  ("unit", 'r', False),

    # load  [l]
    "maxLinkUtilizationAsEdgeDst":              ("unit", 'l', False),
    "avgLinkUtilizationAsEdgeDst":              ("unit", 'l', False),
    "maxTxQueueRelativeLoadAsEdgeDst":          ("unit", 'l', False),
    "avgTxQueueRelativeLoadAsEdgeDst":          ("unit", 'l', False),
}

node_feat_names = list(ALL_NODE_FEATURES.keys())
node_feat_in_struct_order = [feat for feat in observed_node_dtype.names if feat in node_feat_names]


ALL_EDGE_FEATURES = {

    # spec  [s]
    "channelDataRate":              ("large", 's'),
    "channelDelay":                 ("large", 's'),
    "txQueueCapacity":              ("large", 's'),

    # channel  [c]
    "transmitted":                  ("large", 'c'),
    "transmittable":                ("large", 'c'),
    "macTxDropped":                 ("large", 'c'),
    "phyRxDropped":                 ("large", 'c'),

    # load  [l]
    "txQueueLoad":                  ("large", 'l'),
    "linkUtilization":              ( "unit", 'l'),
    "txQueueRelativeLoad":          ( "unit", 'l'),
}

edge_feat_names = list(ALL_EDGE_FEATURES.keys())
edge_feat_in_struct_order = [feat for feat in observed_edge_dtype.names if feat in edge_feat_names]
edge_datarate_idx = edge_feat_in_struct_order.index("channelDataRate")
edge_delay_idx = edge_feat_in_struct_order.index("channelDelay")
edge_lu_idx = edge_feat_in_struct_order.index("linkUtilization")
edge_queue_rel_load_idx = edge_feat_in_struct_order.index("txQueueRelativeLoad")


def get_acceptable_features(feat_config: DictConfig):
    """
    Returns those features for the given config that shall be used by the policy.
    """
    global_features = [feat_name for feat_name in global_feat_in_struct_order
                       if ALL_GLOBAL_FEATURES[feat_name][1] in feat_config.global_features]

    node_features = [feat_name for feat_name in node_feat_in_struct_order
                     if ALL_NODE_FEATURES[feat_name][1] in feat_config.node_features
                     and (not feat_config.node_ego_features_only or ALL_NODE_FEATURES[feat_name][2])]

    edge_features = [feat_name for feat_name in edge_feat_in_struct_order
                     if ALL_EDGE_FEATURES[feat_name][1] in feat_config.edge_features]

    return {
        "global": global_features,
        "node": node_features,
        "edge": edge_features
    }
