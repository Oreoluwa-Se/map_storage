#ifndef MAP_HUB_HPP
#define MAP_HUB_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/sections/worker_pool.hpp"
#include "map_storage/utils/alias.hpp"

template <typename T>
struct Config
{
    /*
     * Manage the kd tree also holds the root node.
     */
    using Ptr = std::shared_ptr<Config<T>>;

    Config(
        int max_points_in_vox = -1,
        size_t max_points_in_oct_layer = 30,
        T imbal_factor = 0.63, T del_nodes_factor = 0.5,
        bool track_stats = false,
        size_t init_map_size = std::numeric_limits<size_t>::max(),
        T voxel_size = 1.0);

    bool insert_or_create_block(const Eigen::Vector3i &vox, BlockPtr<T> &new_b);

    void set_root(const BlockPtr<T> &_root);

    void set_support_info(PoolPtr &wp_ptr);

    BlockPtr<T> get_root();

    template <typename PointContainer>
    void grouping_points(PointContainer &build_points, BlockPtrVecCC<T> &voxels);

    BlockPtr<T> search(const Eigen::Vector3i &vox);

    void replace_voxel(BlockPtr<T> &new_blk);

    void erase(BlockPtr<T> &new_blk);

    std::pair<size_t, VisualizationPointStorage<T>> map_points();

    size_t get_voxel_map_size();

public:
    int max_points_in_vox;
    size_t max_points_in_oct_layer;
    T imbal_factor, del_nodes_factor;
    bool track_stats;
    size_t init_map_size;
    T voxel_size;
    BlockPtr<T> root;
    PoolPtr wp = nullptr;

private:
    BlockPtrMap<T> block_map;
    mutable boost::shared_mutex root_mtx, map_mtx;
};

template <typename T>
using ConfigPtr = typename Config<T>::Ptr;
#endif