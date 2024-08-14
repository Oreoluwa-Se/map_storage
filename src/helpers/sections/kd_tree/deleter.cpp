#include "map_storage/sections/kd_tree/deleter.hpp"
#include <tbb/parallel_invoke.h>

template <typename T>
void Deleter<T>::set_support_info(ConfigPtr<T> &config_ptr)
{
    config = config_ptr;
}

template <typename T>
void Deleter<T>::delete_within_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type)
{
    if (BlockPtr<T> root = config->get_root())
        range_delete(root, ptd, range, DeleteCondition::Inside, del_type);
}

template <typename T>
void Deleter<T>::delete_outside_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type)
{
    if (BlockPtr<T> root = config->get_root())
        range_delete(root, ptd, range, DeleteCondition::Outside, del_type);
}

template <typename T>
void Deleter<T>::handle_collapse(BlockPtr<T> c_block, DeleteCondition cond)
{
    if (!c_block)
        return;

    std::vector<BlockPtr<T>> visit_stack;
    visit_stack.push_back(c_block);

    while (!visit_stack.empty())
    {
        auto node = visit_stack.back();
        visit_stack.pop_back();

        if (!node)
            continue;

        // Check if it is a leaf node.
        BlockPtr<T> left_blk, right_blk;
        if (!node->is_leaf(left_blk, right_blk))
        {
            if (left_blk)
                visit_stack.push_back(left_blk);

            if (right_blk)
                visit_stack.push_back(right_blk);
        }

        if (cond == DeleteCondition::Outside)
            node->set_status(NodeStatus::Deleted);
        else
        {
            // ....... clearing node stuff .......
            config->erase(node); // remove from tracking list
            node->oct->clear();  // clear oct nodes
            node->oct.reset();   // clear octree
            node.reset();        // clear block
        }
    }
}

template <typename T>
void Deleter<T>::range_delete(BlockPtr<T> &c_block, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type)
{
    if (!c_block)
        return;

    if (c_block->get_status() == NodeStatus::Rebalancing)
    {
        c_block->log_delete(center, range, cond, del_type);
        return;
    }

    MinMaxHolder<T> bbox_info = c_block->get_min_max();
    auto del_status = DeleteManager<T>::skip_criteria(cond, del_type, bbox_info, center, range);
    if (del_status == DeleteStatus::Skip)
    {
        // update subtree information
        c_block->update_subtree_info();
        return;
    }

    // here we collapse the subtree
    if (del_status == DeleteStatus::Collapse)
    {
        // detach from parent
        if (auto sp = c_block->get_aux_connection(Connection::Parent))
        {
            auto left_child = sp->get_child(Connection::Left);
            if (left_child && left_child->node_rep == c_block->node_rep)
                sp->set_child(nullptr, Connection::Left);
            else
                sp->set_child(nullptr, Connection::Right);
        }

        // handling the node connection later
        std::weak_ptr<Deleter<T>> workptr = this->shared_from_this();
        config->wp->enqueue_task(
            TaskType::CollapseUnwanted,
            [workptr, c_block, cond]()
            {
                if (auto sp = workptr.lock())
                    sp->handle_collapse(c_block, cond);
            },
            PriorityRank::Medium);

        // update subtree information
        c_block->set_status(NodeStatus::Deleted);
        c_block->update_subtree_info();
        return;
    }

    // performs actual delete operation
    if (cond == DeleteCondition::Outside)
        c_block->oct->outside_range_delete(center, range, del_type);
    else if (cond == DeleteCondition::Inside)
        c_block->oct->within_range_delete(center, range, del_type);

    // moving left or right or both
    BlockPtr<T> left_blk, right_blk;
    if (c_block->is_leaf(left_blk, right_blk))
    {
        tbb::parallel_invoke(
            [&]
            { range_delete(left_blk, center, range, cond, del_type); },
            [&]
            { range_delete(right_blk, center, range, cond, del_type); });
    }
    else
    {
        if (left_blk)
            range_delete(left_blk, center, range, cond, del_type);

        if (right_blk)
            range_delete(right_blk, center, range, cond, del_type);
    }

    // mark current block as deleted if the criteria calls for that
    if (c_block->oct->bbox->get_size() == 0)
        c_block->set_status(NodeStatus::Deleted);

    // update subtree information
    c_block->update_subtree_info();
}

template struct Deleter<double>;
template struct Deleter<float>;