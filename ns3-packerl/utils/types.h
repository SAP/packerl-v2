#ifndef NS3_PACKERL_TYPES_H
#define NS3_PACKERL_TYPES_H

#include <functional>
#include <tuple>
#include <utility>

#include "ns3/core-module.h"
#include "ns3/network-module.h"


using std::pair, std::hash, std::tuple, std::unordered_map;

typedef pair<uint32_t, uint32_t> U32Pair;
typedef pair<uint64_t, uint64_t> U64Pair;

typedef unordered_map<uint32_t, uint32_t> U32ToU32Map;
typedef unordered_map<uint32_t, uint64_t> U32ToU64Map;
typedef unordered_map<uint32_t, double> U32ToDMap;
typedef unordered_map<uint32_t, bool> U32ToBoolMap;

typedef unordered_map<uint32_t, U32ToU32Map> U32ToU32ToU32Map;
typedef unordered_map<uint32_t, U32ToU64Map> U32ToU32ToU64Map;
typedef unordered_map<uint32_t, U32ToDMap> U32ToU32ToDMap;
typedef unordered_map<uint32_t, U32ToBoolMap> U32ToU32ToBoolMap;

/**
 * Hash function for pairs.
 * @tparam T1 first element
 * @tparam T2 second element
 */
struct pair_hash
{
    template <class T1, class T2>
    size_t operator() (const pair<T1, T2> &pair) const
    {
        auto hash1 = hash<T1>()(pair.first);
        auto hash2 = hash<T2>()(pair.second);
        return hash1 == hash2 ? hash1 : hash1 ^ hash2;
    }
};

typedef unordered_map<U32Pair, uint32_t, pair_hash> U32PairToU32Map;
typedef unordered_map<U32Pair, uint64_t, pair_hash> U32PairToU64Map;
typedef unordered_map<U32Pair, double, pair_hash> U32PairToDMap;
typedef unordered_map<U32Pair, bool, pair_hash> U32PairToBoolMap;
typedef unordered_map<U32Pair, U32Pair, pair_hash> U32PairToU32PairMap;

typedef tuple<uint32_t, uint32_t, uint32_t> U32Triple;

/**
 * Hash function for triples.
 */
struct triple_hash
{
    template <class T1, class T2, class T3>
    size_t operator() (const std::tuple<T1, T2, T3> &triple) const
    {
        auto hash1 = std::hash<T1>()(std::get<0>(triple));
        auto hash2 = std::hash<T2>()(std::get<1>(triple));
        auto hash3 = std::hash<T3>()(std::get<2>(triple));

        // use boost's hash_combine with reciprocal of golden ratio as magic number (TODO is there a better way?)
        size_t seed = hash1;
        seed ^= hash2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hash3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
};

typedef unordered_map<U32Triple, uint32_t, triple_hash> U32TripleToU32Map;
typedef unordered_map<U32Triple, uint64_t, triple_hash> U32TripleToU64Map;

template <typename MapType, typename KeyType, typename ValueType>
ValueType getValueOrDefault(const MapType& map, const KeyType& key, const ValueType& defaultValue)
{
    auto it = map.find(key);
    if (it != map.end()) {
        return it->second;
    }
    return defaultValue;
}


#endif // NS3_PACKERL_TYPES_H
