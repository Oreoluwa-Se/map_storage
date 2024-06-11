#include "map_storage/sections/kd_tree/maphub.hpp"
#include <tbb/parallel_for_each.h>

template <typename T>
Config<T>::Config(
    size_t max_points_in_vox, T imbal_factor, T del_nodes_factor,
    bool track_stats, size_t init_map_size, T voxel_size)
    : max_points_in_vox(max_points_in_vox),
      imbal_factor(imbal_factor),
      del_nodes_factor(del_nodes_factor),
      track_stats(track_stats),
      init_map_size(init_map_size), voxel_size(voxel_size) {}

template <typename T>
bool Config<T>::insert_or_create_block(const Eigen::Vector3i &vox, BlockPtr<T> &new_b)
{
    /* When given a voxel it either pulls from the currently tracked list or creates one */
    typename BlockPtrMap<T>::accessor loc;
    if (block_map.insert(loc, vox))
    {
        new_b = std::make_shared<Block<T>>(vox, max_points_in_vox, voxel_size, track_stats);
        loc->second = new_b;
        return true;
    }

    // retrieve existing block
    new_b = loc->second;
    return false;
}

template <typename T>
void Config<T>::set_root(const BlockPtr<T> &_root)
{
    boost::unique_lock<boost::shared_mutex> lock(root_mtx);
    root = _root;
}

template <typename T>
BlockPtr<T> Config<T>::get_root()
{
    boost::shared_lock<boost::shared_mutex> lock(root_mtx);
    return root;
}

template <typename T>
void Config<T>::set_support_info(PoolPtr &wp_ptr)
{
    wp = wp_ptr;
}

template <typename T>
template <typename PointContainer>
void Config<T>::grouping_points(PointContainer &build_points, BlockPtrVecCC<T> &voxels)
{
    voxels.reserve(0.5 * build_points.size());

    tbb::parallel_for_each(
        build_points.begin(), build_points.end(),
        [&](auto &point)
        {
            BlockPtr<T> block = nullptr;

            if (insert_or_create_block(point->vox, block))
                voxels.emplace_back(block);

            // insert into voxel
            block->oct->split_insert_point(point);
        });

    // flush the remaning points on the worker thread
    auto voxels_ptr = std::make_shared<BlockPtrVecCC<T>>(voxels);
    wp->enqueue_task(
        TaskType::PointInsert,
        [voxels_ptr]()
        {
            for (auto &blk : *voxels_ptr)
                blk->oct->split_batch_insert();
        },
        PriorityRank::Super);
}

template <typename T>
BlockPtr<T> Config<T>::search(const Eigen::Vector3i &vox)
{
    typename BlockPtrMap<T>::accessor accessor;
    if (block_map.find(accessor, vox))
        return accessor->second;

    return nullptr;
}

template <typename T>
void Config<T>::replace_voxel(BlockPtr<T> &new_blk)
{
    typename BlockPtrMap<T>::accessor loc;
    if (block_map.find(loc, new_blk->node_rep))
    {
        // replace when not pointing to similar objects
        if (loc->second != new_blk)
            loc->second = new_blk;
    }
}

// setting templates for compiler
template struct Config<double>;
template struct Config<float>;

template void Config<double>::grouping_points<Point3dPtrVect<double>>(Point3dPtrVect<double> &, BlockPtrVecCC<double> &);
template void Config<double>::grouping_points<Point3dPtrVectCC<double>>(Point3dPtrVectCC<double> &, BlockPtrVecCC<double> &);

template void Config<float>::grouping_points<Point3dPtrVect<float>>(Point3dPtrVect<float> &, BlockPtrVecCC<float> &);
template void Config<float>::grouping_points<Point3dPtrVectCC<float>>(Point3dPtrVectCC<float> &, BlockPtrVecCC<float> &);