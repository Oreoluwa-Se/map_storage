#ifndef MAP_BUILDER_HPP
#define MAP_BUILDER_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/sections/worker_pool.hpp"
#include "map_storage/utils/alias.hpp"
#include "maphub.hpp"

template <typename T>
struct BuildHelper
{
    /*
     * Based off this paper: https://arxiv.org/pdf/1410.5420.pdf
     * we use variance to decide the axis to split on
     */

    void pre_sort(const BlockPtrVecCC<T> &blocks);

    void pre_allocator(BuildHelper<T> &left, BuildHelper<T> &right, int new_size = -1);

    void xyz_split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right);

    void yzx_split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right);

    void zxy_split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right);

    int axis_calc(const BlockPtrVecCC<T> &blocks);

    void split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right, int axis = -1);

    void clear();

    void print();

    // attributes
    BlockPtr<T> med_block;
    int calc_axis = -1;
    std::vector<size_t> xyz, yzx, zxy;
};

// .................... ACTUAL BUILDER ....................
template <typename T>
struct MapBuilder
{
    using Ptr = std::shared_ptr<MapBuilder<T>>;

    void set_support_info(ConfigPtr<T> &config_ptr);

    bool build(Point3dPtrVect<T> &points);

    BlockPtr<T> rebuild(BlockPtrVecCC<T> &vox_blocks);

private:
    BlockPtr<T> build_base(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &handler);

    BlockPtr<T> modify_block(BlockPtr<T> block, int axis, bool set_axis);

private:
    Point3dPtrVect<T> build_points;
    ConfigPtr<T> config = nullptr;
};

template <typename T>
using MapBuilderPtr = typename MapBuilder<T>::Ptr;
#endif