#ifndef REBALANCE_ALGO_HPP
#define REBALANCE_ALGO_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/utils/alias.hpp"
#include "maphub.hpp"
#include "deleter.hpp"
#include "inserter.hpp"
#include "builder.hpp"
#include <memory>
#include <map>

template <typename T>
struct Rebalancer
{
    using Ptr = std::shared_ptr<Rebalancer<T>>;

    void set_support_info(ConfigPtr<T> &config_ptr, InserterPtr<T> &insert, DeleterPtr<T> &delete_ptr, MapBuilderPtr<T> &build_ptr);

    void sequencer(BlockPtrVecCC<T> &scapegoats, std::map<size_t, BlockPtrVec<T>> &run_seq);

    void run_algo(BlockPtr<T> blk);

private:
    BlockPtr<T> rebuild_phase(BlockPtr<T> &blk, BlockPtrVecCC<T> &cloned_blks, BlockPtrVecCC<T> &to_delete);

    void flatten(BlockPtr<T> &blk, BlockPtrVecCC<T> &cloned_blks, BlockPtrVecCC<T> &to_delete);

    void run_operations(OperationLog<T> &log, BlockPtr<T> &new_lead);

    void reattach_block(BlockPtr<T> &old, BlockPtr<T> &new_b);

    void enqueue_block_delete(BlockPtrVecCC<T> &to_delete);

private:
    ConfigPtr<T> config = nullptr;
    InserterPtr<T> inserter = nullptr;
    DeleterPtr<T> deleter = nullptr;
    MapBuilderPtr<T> builder = nullptr;
};

template <typename T>
using RebalancerPtr = typename Rebalancer<T>::Ptr;

#endif