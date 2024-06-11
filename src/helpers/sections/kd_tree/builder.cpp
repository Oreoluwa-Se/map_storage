#include "map_storage/sections/kd_tree/builder.hpp"
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>
#include <iomanip>
#include <tbb/task_group.h>

template <typename T>
void BuildHelper<T>::pre_sort(const BlockPtrVecCC<T> &blocks)
{
    xyz.resize(blocks.size());
    yzx.resize(blocks.size());
    zxy.resize(blocks.size());

    std::iota(xyz.begin(), xyz.end(), 0);
    std::iota(yzx.begin(), yzx.end(), 0);
    std::iota(zxy.begin(), zxy.end(), 0);

    // Lambda to simplify sorting logic
    auto sort_indices = [&](std::vector<size_t> &indices, const std::array<int, 3> &order)
    {
        tbb::parallel_sort(
            indices.begin(), indices.end(),
            [&](size_t i1, size_t i2)
            { return std::tie(blocks[i1]->node_rep_d[order[0]], blocks[i1]->node_rep_d[order[1]], blocks[i1]->node_rep_d[order[2]]) <
                     std::tie(blocks[i2]->node_rep_d[order[0]], blocks[i2]->node_rep_d[order[1]], blocks[i2]->node_rep_d[order[2]]); });
    };

    // Sort based on block coordinates
    sort_indices(xyz, {0, 1, 2}); // xyz order
    sort_indices(yzx, {1, 2, 0}); // yzx order
    sort_indices(zxy, {2, 0, 1}); // zxy order
}

template <typename T>
void BuildHelper<T>::clear()
{
    xyz.clear();
    yzx.clear();
    zxy.clear();
}

template <typename T>
void BuildHelper<T>::print()
{
    // Find maximum widths for each column
    std::vector<int> widths = {0, 0, 0};
    for (size_t i = 0; i < xyz.size(); ++i)
        widths[0] = std::max(widths[0], static_cast<int>(std::to_string(xyz[i]).length()));
    for (size_t i = 0; i < yzx.size(); ++i)
        widths[1] = std::max(widths[1], static_cast<int>(std::to_string(yzx[i]).length()));
    for (size_t i = 0; i < zxy.size(); ++i)
        widths[2] = std::max(widths[2], static_cast<int>(std::to_string(zxy[i]).length()));

    std::ostringstream oss;
    // Print headers with padding
    oss << "\n===============" << std::endl;
    oss << std::left << std::setw(widths[0] + 4) << "XZY"
        << std::left << std::setw(widths[1] + 4) << "YZX"
        << std::left << std::setw(widths[2] + 4) << "ZXY"
        << "\n";

    size_t max_len = std::max(xyz.size(), std::max(yzx.size(), zxy.size()));
    for (size_t i = 0; i < max_len; ++i)
    {
        // Print YZX column with right alignment
        if (i < xyz.size())
            oss << std::right << std::setw(widths[0]) << xyz[i];
        else
            oss << std::setw(widths[0]) << "";
        oss << "    "; // Fixed spacing between columns

        // Print XZY column with right alignment
        if (i < yzx.size())
            oss << std::right << std::setw(widths[1]) << yzx[i];
        else
            oss << std::setw(widths[1]) << "";
        oss << "    "; // Fixed spacing between columns

        // Print ZXY column with right alignment
        if (i < zxy.size())
            oss << std::right << std::setw(widths[2]) << zxy[i];
        else
            oss << std::setw(widths[2]) << "";

        oss << "\n"; // Newline for the next set of elements
    }
    oss << "===============" << std::endl;
    std::cout << oss.str();
}

template <typename T>
void BuildHelper<T>::pre_allocator(BuildHelper<T> &left, BuildHelper<T> &right, int new_size_)
{
    size_t new_size = new_size_ == -1 ? xyz.size() / 2 : new_size_;

    // pre-allocate space left
    left.xyz.reserve(new_size);
    left.yzx.reserve(new_size);
    left.zxy.reserve(new_size);

    // pre-allocate space right
    right.xyz.reserve(new_size);
    right.yzx.reserve(new_size);
    right.zxy.reserve(new_size);
}

template <typename T>
void BuildHelper<T>::xyz_split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right)
{
    /* if axis= 0 we take the median of xyz.
        1) the left side goes into left.xyz and right side goes into right.xyz
        2) From blocks we get the actual xyz value from blocks then we go through the indices
        for yzx and zxy and compare them in xyz order to insert into the left.xyz or right.xyz
        and left.zxy, or right.zxy
    */
    left.clear();
    right.clear();

    int actual_index = xyz[xyz.size() / 2];

    // the pivot block
    med_block = blocks[actual_index];
    auto comp_tup = std::tie(med_block->node_rep_d[0], med_block->node_rep_d[1], med_block->node_rep_d[2]);

    // split indices into relevant sections
    auto mid = xyz.begin() + xyz.size() / 2;
    std::move(xyz.begin(), mid, std::back_inserter(left.xyz));
    std::move(mid + 1, xyz.end(), std::back_inserter(right.xyz));

    left.yzx.reserve(xyz.size() / 2);
    right.yzx.reserve(xyz.size() / 2);

    left.zxy.reserve(yzx.size() / 2);
    right.zxy.reserve(yzx.size() / 2);

    // perform split on others
    for (size_t idx = 0; idx < yzx.size(); ++idx)
    {
        if (yzx[idx] != actual_index)
        {
            if (comp_tup > std::tie(blocks[yzx[idx]]->node_rep_d[0], blocks[yzx[idx]]->node_rep_d[1], blocks[yzx[idx]]->node_rep_d[2]))
                left.yzx.push_back(yzx[idx]);
            else
                right.yzx.push_back(yzx[idx]);
        }

        if (zxy[idx] != actual_index)
        {
            if (comp_tup > std::tie(blocks[zxy[idx]]->node_rep_d[0], blocks[zxy[idx]]->node_rep_d[1], blocks[zxy[idx]]->node_rep_d[2]))
                left.zxy.push_back(zxy[idx]);
            else
                right.zxy.push_back(zxy[idx]);
        }
    }
}

template <typename T>
void BuildHelper<T>::yzx_split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right)
{
    /* if axis= 1 we take the median of yzx.
        1) the left side goes into left.yzx and right side goes into right.yzx
        2) From blocks we get the actual xyz value from blocks then we go through the indices
        for xyz and zxy and compare them in yzx order to insert into the left.xyz or right.xyz
        and left.zxy, or right.zxy
    */
    left.clear();
    right.clear();

    int actual_index = yzx[yzx.size() / 2];
    // the pivot block
    med_block = blocks[actual_index];
    auto comp_tup = std::tie(med_block->node_rep_d[1], med_block->node_rep_d[2], med_block->node_rep_d[0]);

    // split indices into relevant sections
    auto mid = yzx.begin() + yzx.size() / 2;
    std::move(yzx.begin(), mid, std::back_inserter(left.yzx));
    std::move(mid + 1, yzx.end(), std::back_inserter(right.yzx));

    left.xyz.reserve(xyz.size() / 2);
    right.xyz.reserve(xyz.size() / 2);

    left.zxy.reserve(yzx.size() / 2);
    right.zxy.reserve(yzx.size() / 2);

    // perform split on others
    for (size_t idx = 0; idx < yzx.size(); ++idx)
    {
        if (xyz[idx] != actual_index)
        {
            if (comp_tup > std::tie(blocks[xyz[idx]]->node_rep_d[1], blocks[xyz[idx]]->node_rep_d[2], blocks[xyz[idx]]->node_rep_d[0]))
                left.xyz.push_back(xyz[idx]);
            else
                right.xyz.push_back(xyz[idx]);
        }

        if (zxy[idx] != actual_index)
        {
            if (comp_tup > std::tie(blocks[zxy[idx]]->node_rep_d[1], blocks[zxy[idx]]->node_rep_d[2], blocks[zxy[idx]]->node_rep_d[0]))
                left.zxy.push_back(zxy[idx]);
            else
                right.zxy.push_back(zxy[idx]);
        }
    }
}

template <typename T>
void BuildHelper<T>::zxy_split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right)
{
    left.clear();
    right.clear();

    const int actual_index = zxy[zxy.size() / 2];
    med_block = blocks[actual_index];
    auto comp_tup = std::tie(med_block->node_rep_d[2], med_block->node_rep_d[0], med_block->node_rep_d[1]);

    // split indices into relevant sections
    auto mid = zxy.begin() + zxy.size() / 2;
    std::move(zxy.begin(), mid, std::back_inserter(left.zxy));
    std::move(mid + 1, zxy.end(), std::back_inserter(right.zxy));

    left.xyz.reserve(xyz.size() / 2);
    right.xyz.reserve(xyz.size() / 2);

    left.yzx.reserve(yzx.size() / 2);
    right.yzx.reserve(yzx.size() / 2);

    // perform split on others
    for (size_t idx = 0; idx < xyz.size(); ++idx)
    {
        if (xyz[idx] != actual_index)
        {
            if (comp_tup > std::tie(blocks[xyz[idx]]->node_rep_d[2], blocks[xyz[idx]]->node_rep_d[0], blocks[xyz[idx]]->node_rep_d[1]))
                left.xyz.push_back(xyz[idx]);
            else
                right.xyz.push_back(xyz[idx]);
        }

        if (yzx[idx] != actual_index)
        {
            if (comp_tup > std::tie(blocks[yzx[idx]]->node_rep_d[2], blocks[yzx[idx]]->node_rep_d[0], blocks[yzx[idx]]->node_rep_d[1]))
                left.yzx.push_back(yzx[idx]);
            else
                right.yzx.push_back(yzx[idx]);
        }
    }
}

template <typename T>
int BuildHelper<T>::axis_calc(const BlockPtrVecCC<T> &blocks)
{
    if (xyz.empty())
        return -1;

    Eigen::Matrix<T, 3, 1> sum = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 1> sum_sq = Eigen::Matrix<T, 3, 1>::Zero();

    for (const auto &val : xyz)
    {
        const Eigen::Matrix<T, 3, 1> &val_m = blocks[val]->node_rep_d;
        sum.noalias() += val_m;
        sum_sq.noalias() += val_m.cwiseProduct(val_m);
    }

    T n = static_cast<T>(xyz.size());
    Eigen::Matrix<T, 3, 1> mean = sum / n;
    Eigen::Matrix<T, 3, 1> var = (sum_sq / n) - mean.cwiseProduct(mean);

    calc_axis = 0; // 0 for X, 1 for Y, 2 for Z
    if (var(1) > var(calc_axis))
        calc_axis = 1;
    if (var(2) > var(calc_axis))
        calc_axis = 2;

    return calc_axis;
}

template <typename T>
void BuildHelper<T>::split(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &left, BuildHelper<T> &right, int axis)
{
    int new_size = xyz.size() / 2;

    // space allocator
    pre_allocator(left, right, new_size);

    // calculate axis if not givex
    if (axis == -1)
        axis = axis_calc(blocks);

    /* and similar when axis=2*/
    switch (axis)
    {
    case 0: // Splitting on xyz
        xyz_split(blocks, left, right);
        break;
    case 1: // Splitting on yzx
        yzx_split(blocks, left, right);
        break;
    case 2: // Splitting on zxy
        zxy_split(blocks, left, right);
        break;
    }
}

template struct BuildHelper<double>;
template struct BuildHelper<float>;

// .................... ACTUAL BUILDER ....................
template <typename T>
void MapBuilder<T>::set_support_info(ConfigPtr<T> &config_ptr)
{
    config = config_ptr;
}

template <typename T>
bool MapBuilder<T>::build(Point3dPtrVect<T> &points)
{
    if (points.empty())
        return false;

    std::move(points.begin(), points.end(), std::back_inserter(build_points));
    if (config->init_map_size != std::numeric_limits<size_t>::max())
    {
        // we don't build unless we have maximum point
        if (build_points.size() < config->init_map_size)
            return false;
    }

    BlockPtrVecCC<T> voxels;
    config->grouping_points(build_points, voxels);

    // pre sort each index
    BuildHelper<T> vec_handler;
    vec_handler.pre_sort(voxels);

    // build tree and call pullup
    BlockPtr<T> root = build_base(voxels, vec_handler);
    // set root in config
    root->update_subtree_info();
    config->set_root(root);

    // clear points and return true to indicate we're done
    build_points.clear();

    return true;
}

template <typename T>
BlockPtr<T> MapBuilder<T>::modify_block(BlockPtr<T> block, int axis, bool set_axis)
{
    if (set_axis)
        block->set_axis(axis);

    block->set_status(NodeStatus::Connected);
    return block;
}

template <typename T>
BlockPtr<T> MapBuilder<T>::build_base(const BlockPtrVecCC<T> &blocks, BuildHelper<T> &handler)
{
    if (handler.xyz.empty())
        return nullptr;

    if (handler.xyz.size() == 1)
    {
        // Handle the base case where only one block is present
        return modify_block(blocks[handler.xyz[0]], 0, false);
    }

    BuildHelper<T> left, right;
    handler.split(blocks, left, right);

    // Modyfy and extract the selected block
    auto block = modify_block(handler.med_block, handler.calc_axis, true);

    auto run_task = [&](BuildHelper<T> &hand, bool left_side)
    {
        auto child = build_base(blocks, hand);
        if (child)
        {
            child->set_aux_connection(block, Connection::Parent);
            block->set_child(child, (left_side) ? Connection::Left : Connection::Right);
        }
    };

    if (!left.xyz.empty() && !right.xyz.empty())
    {
        tbb::task_group group;
        group.run(
            [&]
            { run_task(left, true); });
        group.run(
            [&]
            { run_task(right, false); });

        group.wait();
    }
    else if (!left.xyz.empty())
        run_task(left, true);

    else if (!right.xyz.empty())
        run_task(right, false);

    if (block) // updates height information e.t.c
        block->update_subtree_info();

    return block;
}

template <typename T>
BlockPtr<T> MapBuilder<T>::rebuild(BlockPtrVecCC<T> &vox_blocks)
{
    if (vox_blocks.empty())
        return nullptr;

    BuildHelper<T> vec_handler;
    vec_handler.pre_sort(vox_blocks);

    // build tree and call pullup
    BlockPtr<T> base = build_base(vox_blocks, vec_handler);

    if (base)
        base->update_subtree_info();

    return base;
}

template struct MapBuilder<double>;
template struct MapBuilder<float>;