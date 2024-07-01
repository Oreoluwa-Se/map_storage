#include "map_storage/sections/kd_tree/searcher.hpp"
#include <iomanip>
#include <sstream>

template <typename T>
SearchRunner<T>::SearchRunner(const Eigen::Matrix<T, 3, 1> &query_point, T vox_size, size_t num_nearest, T &max_range, SearchType typ)
    : node_rep(Point3d<T>::calc_vox_index(query_point, vox_size)),
      node_rep_d(node_rep.cast<T>()), qp(query_point),
      voxel_size(vox_size),
      num_nearest(num_nearest), max_range(max_range),
      max_range_sq(max_range * max_range), typ(typ) {}

template <typename T>
void SearchRunner<T>::mark(const Eigen::Vector3i &vox)
{
    typename BlockTrackType::accessor n_loc;
    if (filter.insert(n_loc, vox))
        n_loc->second = true;
}

// used to avoid duplicate operations
template <typename T>
bool SearchRunner<T>::not_visited(const Eigen::Vector3i &vox) const
{
    typename BlockTrackType::accessor n_loc;
    if (filter.find(n_loc, vox))
        return false;

    return true;
}

template <typename T>
void SearchRunner<T>::update_search_bucket(BlockPtr<T> &block, T sq_dist)
{
    if (typ == SearchType::Distribution)
    {
        if (block->oct->bbox->get_size() < 3)
            return;

        handle_distributive(block, sq_dist);
        return;
    }

    handle_point(block);
}

template <typename T>
void SearchRunner<T>::handle_distributive(BlockPtr<T> &block, T sq_dist)
{
    if (blocks.size() < num_nearest || sq_dist < max_range_sq)
    {

        blocks.push({sq_dist, block});
        if (blocks.size() > num_nearest)
            blocks.pop();
    }

    if (blocks.size() == num_nearest && blocks.top().first < max_range_sq)
    {
        max_range = std::sqrt(blocks.top().first);
        max_range_sq = blocks.top().first;
    }
}

template <typename T>
void SearchRunner<T>::enqueue_next_distributive(ExplorePriorityType &next_block, BlockPtr<T> &curr)
{
    BlockPtr<T> left_blk, right_blk;
    if (curr->is_leaf(left_blk, right_blk))
        return;

    Eigen::Matrix<T, 3, 1> mean = curr->oct->bbox->get_mean();
    int axis = curr->get_axis();

    bool move_left = qp[axis] < mean[axis];
    BlockPtr<T> closer = move_left ? left_blk : right_blk;
    BlockPtr<T> further = move_left ? right_blk : left_blk;

    if (closer && not_visited(closer->node_rep))
    {
        mean = closer->oct->bbox->get_mean();
        auto sq_dist = (mean - qp).squaredNorm();
        next_block.push({sq_dist, closer});
    }

    if (further && not_visited(further->node_rep))
    {
        mean = further->oct->bbox->get_mean();
        if (blocks.size() < num_nearest || std::pow(qp[axis] - mean[axis], 2) <= max_range_sq)
        {
            auto sq_dist = (mean - qp).squaredNorm();
            next_block.push({sq_dist, further});
        }
    }
}

template <typename T>
void SearchRunner<T>::handle_point(BlockPtr<T> &block)
{
    block->oct->radius_search(points, qp, max_range, num_nearest);
    max_range_sq = max_range * max_range;
}

template <typename T>
void SearchRunner<T>::enqueue_next_point(ExplorePriorityType &next_block, BlockPtr<T> &curr)
{
    BlockPtr<T> left_blk, right_blk;
    if (curr->is_leaf(left_blk, right_blk))
        return;

    for (auto &child : {left_blk, right_blk})
    {
        if (child)
        {
            if (not_visited(child->node_rep))
            {
                T child_dist = child->closest_distance(qp);
                if (points.size() < num_nearest || child_dist <= max_range_sq)
                    next_block.push({child_dist, child});
            }
        }
    }
}

template <typename T>
void SearchRunner<T>::enqueue_next(ExplorePriorityType &next_block, BlockPtr<T> &curr)
{

    if (typ == SearchType::Distribution)
    {
        enqueue_next_distributive(next_block, curr);
        return;
    }

    enqueue_next_point(next_block, curr);
}

template struct SearchRunner<double>;
template struct SearchRunner<float>;

template <typename T>
void Searcher<T>::set_support_info(ConfigPtr<T> &config_ptr)
{
    config = config_ptr;
}

template <typename T>
void Searcher<T>::explore_tree(SearchRunner<T> &opt)
{
    if (opt.max_range <= 0)
        return;

    typename SearchRunner<T>::ExplorePriorityType next_block;

    // Anchor points to speed up search criteria
    if (auto anchor_point = config->search(opt.node_rep))
        next_block.emplace(anchor_point->closest_distance(opt.qp), anchor_point);

    T half_voxel = 0.5 * opt.voxel_size;
    for (int i = 0; i < 8; ++i)
    {
        Eigen::Matrix<T, 3, 1> variation = opt.qp;
        variation.x() += (i & 1) ? half_voxel : -half_voxel;
        variation.y() += (i & 2) ? half_voxel : -half_voxel;
        variation.z() += (i & 4) ? half_voxel : -half_voxel;

        // include valid options
        Eigen::Vector3i ap = Point3d<T>::calc_vox_index(variation, opt.voxel_size);
        if (auto anchor_point = config->search(ap))
            next_block.emplace(anchor_point->closest_distance(variation), anchor_point);
    }

    // include root as backup case
    auto head_point = config->get_root();
    next_block.emplace(head_point->closest_distance(opt.qp), head_point);

    while (!next_block.empty())
    {
        auto top = next_block.top();
        T sq_dist = top.first;
        auto node = top.second;
        next_block.pop();

        // if node doesn't exist or has already been visited
        if (!node || !opt.not_visited(node->node_rep))
            continue;

        opt.mark(node->node_rep);

        // sq_dist update
        if (opt.typ == SearchType::Distribution)
            sq_dist = (opt.qp - node->oct->bbox->get_mean()).squaredNorm();

        // update the result bucket
        opt.update_search_bucket(node, sq_dist);

        // enque only when within
        opt.enqueue_next(next_block, node);
    }
}

template <typename T>
BlockPtrVec<T> Searcher<T>::get_matched_blocks(SearchRunner<T> &opt, bool verbose)
{
    BlockPtrVec<T> matches;
    matches.reserve(opt.blocks.size());
    std::ostringstream oss;

    if (verbose)
    {
        oss << std::fixed << std::setprecision(4); // Set fixed-point notation and precision
        oss << "Query Distribution: " << Point3d<T>::eig_to_string(opt.qp) << "\n";
        oss << "Matched Distribution:\n";
        oss << std::setw(10) << std::left << "Index" << std::setw(40) << "Mean" << std::setw(15) << "Distance" << "\n";
        oss << "--------------------------------------------------------------\n";
    }

    int index = 1;
    while (!opt.blocks.empty())
    {
        auto match = opt.blocks.top();
        opt.blocks.pop();

        matches.push_back(match.second);

        if (verbose)
        {
            auto blk = match.second;
            T distance = std::sqrt(match.first); // Assuming the stored distance is squared
            oss << std::setw(10) << std::left << index
                << std::setw(40) << Point3d<T>::eig_to_string(blk->oct->bbox->get_mean())
                << std::setw(15) << distance << "\n";
        }
        ++index;
    }

    if (verbose)
        std::cout << oss.str();

    return matches;
}

template <typename T>
AVector3TVec<T> Searcher<T>::get_matched_points(SearchRunner<T> &result, bool verbose)
{
    return OctreeNode<T>::get_matched(result.points, result.qp, verbose);
}

template <typename T>
std::vector<BlockPointPair<T>> Searcher<T>::get_matched_points_with_block(SearchRunner<T> &opt, ConfigPtr<T> &config, bool verbose)
{
    std::vector<BlockPointPair<T>> matches;
    matches.reserve(opt.points.size());
    std::ostringstream oss;

    if (verbose)
    {
        oss << std::fixed << std::setprecision(4); // Set fixed-point notation and precision
        oss << "\nQuery Mean: " << Point3d<T>::eig_to_string(opt.qp) << "\n";
        oss << "Matched Distribution Mean:\n";
        oss << std::setw(10) << std::left << "Index"
            << std::setw(40) << "Mean"
            << std::setw(20) << "Voxel"
            << std::setw(15) << "Distance" << "\n";
        oss << "--------------------------------------------------------------\n";
    }

    int index = 1;
    while (!opt.blocks.empty())
    {
        auto match = opt.blocks.top();
        opt.blocks.pop();

        auto blk = match.second;
        Eigen::Matrix<T, 3, 1> point = blk->oct->bbox->get_mean();
        matches.push_back({blk, point});

        if (verbose)
        {
            T distance = std::sqrt(match.first);
            oss << std::setw(10) << std::left << index
                << std::setw(40) << Point3d<T>::eig_to_string(point)
                << std::setw(20) << Point3d<T>::eig_to_string(blk->node_rep)
                << std::setw(15) << distance << "\n";
        }
        ++index;
    }

    if (verbose)
        std::cout << oss.str();

    return matches;
}

template struct Searcher<double>;
template struct Searcher<float>;