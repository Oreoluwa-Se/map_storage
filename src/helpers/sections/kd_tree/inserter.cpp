#include "map_storage/sections/kd_tree/inserter.hpp"
#include <tbb/parallel_for_each.h>

template <typename T>
void Inserter<T>::set_support_info(ConfigPtr<T> &config_ptr)
{
    config = config_ptr;
}

template <typename T>
void Inserter<T>::insert(Point3dPtrVect<T> &points, BlockPtrVecCC<T> &scapegoats)
{
    if (points.empty())
        return;

    BlockPtrVecCC<T> voxels; // voxels contains blocks to be inserted
    config->grouping_points(points, voxels);

    // sets up for for collecting possible scapegoats
    tbb::parallel_for_each(
        voxels.begin(), voxels.end(),
        [&](auto &blk)
        {
            // here means we are inserting a new block
            auto root = config->get_root();
            insertion_handler(root, blk, scapegoats);
        });
}

template <typename T>
void Inserter<T>::insert(Point3dPtrVectCC<T> &points, BlockPtrVecCC<T> &scapegoats)
{

    if (points.empty())
        return;

    BlockPtrVecCC<T> voxels; // voxels contains blocks to be inserted
    config->grouping_points(points, voxels);

    // sets up for for collecting possible scapegoats
    tbb::parallel_for_each(
        voxels.begin(), voxels.end(),
        [&](auto &blk)
        {
            auto root = config->get_root();
            insertion_handler(root, blk, scapegoats);
        });
}

template <typename T>
void Inserter<T>::insertion_handler(BlockPtr<T> &start_point, BlockPtr<T> &to_insert, BlockPtrVecCC<T> &scapegoats)
{
    BlockPtr<T> curr = start_point;
    bool scapegoat_found = false, swapped = false;
    RunningStats<T> r_stats;

    while (curr)
    {
        if (!swapped)
        {
            // essentially is remapping operation has occured we re-direct efforts
            if (auto sp = curr->get_aux_connection(Connection::Remap))
            {
                curr = sp;
                swapped = true;
            }
        }

        // preparation for calculating new_axis
        r_stats.add_info(curr->node_rep_d);

        // decides where we move to next and if we are a sca
        if (auto next_node = curr->insert_or_move(to_insert, r_stats))
        {
            if (!scapegoat_found && curr->scapegoat_handler(config->imbal_factor, config->del_nodes_factor))
            {
                scapegoats.push_back(curr);
                scapegoat_found = true;
            }

            // moving to next
            curr = next_node;
        }
        else
        {
            to_insert->set_aux_connection(curr, Connection::Parent);
            to_insert->set_status(NodeStatus::Connected);
            break;
        }
    }
}

template <typename T>
void Inserter<T>::insertion_cont(BlockPtr<T> &start_point, BlockPtr<T> &block, RunningStats<T> &r_stats)
{
    // quick insertion into the new tree.
    BlockPtr<T> curr = start_point;
    while (curr)
    {
        r_stats.add_info(curr->node_rep_d);

        if (auto next_move = curr->insert_or_move(block, r_stats))
            curr = next_move;
        else
        {
            block->set_aux_connection(curr, Connection::Parent);
            block->set_status(NodeStatus::Connected);
            break;
        }
    }
}

template struct Inserter<double>;
template struct Inserter<float>;