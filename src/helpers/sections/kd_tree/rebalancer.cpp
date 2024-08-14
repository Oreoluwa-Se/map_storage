#include "map_storage/sections/kd_tree/rebalancer.hpp"

template <typename T>
void Rebalancer<T>::set_support_info(ConfigPtr<T> &config_ptr, InserterPtr<T> &insert, DeleterPtr<T> &delete_ptr, MapBuilderPtr<T> &build_ptr)
{
    config = config_ptr;
    inserter = insert;
    deleter = delete_ptr;
    builder = build_ptr;
}

template <typename T>
void Rebalancer<T>::sequencer(BlockPtrVecCC<T> &scapegoats, std::map<size_t, BlockPtrVec<T>> &run_seq)
{
    //  This section set's the order attempts
    for (auto &blk : scapegoats)
    {
        if (!blk)
            continue;

        if (blk->cloned_status())
            continue;

        // This check implies no current rebuilding operation ongoing
        run_seq[blk->get_tree_size()].emplace_back(blk);
    }
}

template <typename T>
void Rebalancer<T>::flatten(BlockPtr<T> &blk, BlockPtrVecCC<T> &cloned_blks, BlockPtrVecCC<T> &to_delete)
{
    if (!blk)
        return;

    std::vector<BlockPtr<T>> visit_stack;
    visit_stack.push_back(blk);

    // trigger to start logging operations
    blk->set_status(NodeStatus::Rebalancing);

    while (!visit_stack.empty())
    {
        auto node = visit_stack.back();
        visit_stack.pop_back();

        if (!node || node->cloned_status())
            continue;

        if (NodeStatus::Deleted != node->get_status())
        {
            auto n_block = node->clone_block();
            n_block->set_aux_connection(node, Connection::Previous);

            // for updating tree and tracking stuff
            cloned_blks.emplace_back(n_block);
        }
        else
            config->erase(node); // removes element from the tracking stuff

        // list of blocks to be deleted
        to_delete.push_back(node);

        // Check if it is a leaf node.
        BlockPtr<T> left_blk, right_blk;
        if (!node->is_leaf(left_blk, right_blk))
        {
            if (left_blk)
                visit_stack.push_back(left_blk);

            if (right_blk)
                visit_stack.push_back(right_blk);
        }
    }
}

template <typename T>
void Rebalancer<T>::run_operations(BlockPtr<T> &op_block, BlockPtr<T> &new_lead)
{
    auto logger = op_block->get_logger();
    if (!logger)
        return;

    auto curr_op = logger->get_operations();
    while (curr_op.second)
    {
        // complete insert operation
        if (curr_op.first == OperationType::Insert)
        {
            if (auto ins_ptr = InsertOp<T>::cast(curr_op.second))
            {
                if (auto blk = config->search(ins_ptr->vox_to_insert))
                {
                    if (blk->cloned_status() || NodeStatus::Connected != blk->get_status())
                        continue;

                    auto n_blk = blk->clone_block();
                    inserter->insertion_cont(new_lead, n_blk, ins_ptr->stats);
                    config->replace_voxel(n_blk);
                }
            }
        }

        // complete delete operation
        if (curr_op.first == OperationType::Delete)
        {
            if (auto del_ptr = DeleteOp<T>::cast(curr_op.second))
                deleter->range_delete(new_lead, del_ptr->point, del_ptr->range, del_ptr->cond, del_ptr->del_type);
        }

        // next viable operation
        curr_op = logger->get_operations();
    }
}

template <typename T>
void Rebalancer<T>::reattach_block(BlockPtr<T> &old, BlockPtr<T> &new_lead)
{
    if (auto parent = old->get_aux_connection(Connection::Parent))
    {
        parent->swap_locations(old, new_lead);
        new_lead->set_aux_connection(parent, Connection::Parent);
    }
    else
        config->set_root(new_lead);
}

template <typename T>
void Rebalancer<T>::run_algo(BlockPtr<T> blk)
{
    if (!blk)
        return;

    // This implies it's being handled already
    if (blk->cloned_status())
        return;

    // rebuild and get new head
    BlockPtrVecCC<T> cloned, to_delete;
    flatten(blk, cloned, to_delete);
    if (auto new_lead = builder->rebuild(cloned))
    {
        // Run preliminary operations that have been logged
        run_operations(blk, new_lead);
        blk->set_aux_connection(new_lead, Connection::Remap);

        for (size_t idx = 0; idx < cloned.size(); ++idx)
            config->replace_voxel(cloned[idx]);

        // run final dangling operations
        run_operations(blk, new_lead);

        // Re-atach block to main
        reattach_block(blk, new_lead);
    }

    if (!to_delete.empty())
        enqueue_block_delete(to_delete);
}

template <typename T>
void Rebalancer<T>::enqueue_block_delete(BlockPtrVecCC<T> &to_delete)
{
    // detach old blocks from current map
    auto voxels_ptr = std::make_shared<BlockPtrVecCC<T>>(to_delete);
    config->wp->enqueue_task(
        TaskType::Cleanup,
        [voxels_ptr]()
        {
            for (auto old_blks : *voxels_ptr)
                old_blks->detach();
        },
        PriorityRank::Min);
}

template struct Rebalancer<double>;
template struct Rebalancer<float>;