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
        if (!blk->rebal_status())
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

    while (!visit_stack.empty())
    {
        auto node = visit_stack.back();
        visit_stack.pop_back();

        if (!node || node->cloned_status())
            continue;

        to_delete.push_back(node); // list of blocks to be deleted
        if (NodeStatus::Deleted != node->get_status())
        {
            auto n_block = node->clone_block();
            n_block->set_aux_connection(node, Connection::Previous);
            cloned_blks.emplace_back(n_block);
        }

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
BlockPtr<T> Rebalancer<T>::rebuild_phase(BlockPtr<T> &blk, BlockPtrVecCC<T> &cloned_blks, BlockPtrVecCC<T> &to_delete)
{
    // rebuild the thing from builder
    flatten(blk, cloned_blks, to_delete);
    return builder->rebuild(cloned_blks);
}

template <typename T>
void Rebalancer<T>::run_operations(OperationLog<T> &log, BlockPtr<T> &new_lead)
{
    for (auto &curr_op : log)
    {
        // complete insert operation
        if (curr_op.first == OperationType::Insert)
        {
            if (auto ins_ptr = InsertOp<T>::cast(curr_op.second))
            {
                auto blk = config->search(ins_ptr->vox_to_insert);
                auto status = blk->get_status();

                // not connected blocks means deleted
                if (status == NodeStatus::Deleted)
                    continue;

                if (NodeStatus::Connected == status)
                {
                    if (blk->cloned_status())
                        continue; // at this stage it's already been cloned
                    else
                        blk = blk->clone_block();

                    inserter->insertion_cont(new_lead, blk, ins_ptr->stats);
                }
            }
        }

        // complete delete operation
        if (curr_op.first == OperationType::Delete)
        {
            if (auto del_ptr = DeleteOp<T>::cast(curr_op.second))
                deleter->range_delete(new_lead, del_ptr->point, del_ptr->range, del_ptr->cond, del_ptr->del_type);
        }
    }
}

template <typename T>
void Rebalancer<T>::reattach_block(BlockPtr<T> &old, BlockPtr<T> &new_lead)
{
    if (auto parent = old->get_aux_connection(Connection::Parent))
    {
        auto left_child = parent->get_child(Connection::Left);
        if (left_child && left_child->node_rep == old->node_rep)
        {
            parent->set_child(new_lead, Connection::Left);
        }
        else
            parent->set_child(new_lead, Connection::Right);

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

    blk->set_status(NodeStatus::Rebalancing);

    // rebuild and get new head
    BlockPtrVecCC<T> cloned, to_delete;
    if (auto new_lead = rebuild_phase(blk, cloned, to_delete))
    {
        // run first set of operations
        if (auto logger = blk->get_logger())
        {
            auto log = logger->get_operations();
            if (!log.empty())
                run_operations(log, new_lead);
        }

        // Remapping previous to current elements
        for (auto &c_blk : cloned)
        {
            config->replace_voxel(c_blk);
            if (auto sp = c_blk->get_aux_connection(Connection::Previous))
                sp->set_aux_connection(new_lead, Connection::Remap);
        }

        // re-connecting
        reattach_block(blk, new_lead);

        // perform outstanding operations
        if (auto logger = blk->get_logger())
        {
            while (true)
            {
                auto log = logger->get_operations();
                if (log.empty())
                    break;
                // run operations
                run_operations(log, new_lead);
            }
        }

        if (!to_delete.empty())
            enqueue_block_delete(to_delete);
    }
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