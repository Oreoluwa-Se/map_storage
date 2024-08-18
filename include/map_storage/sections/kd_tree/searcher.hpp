#ifndef MAP_SEARCHER_HPP
#define MAP_SEARCHER_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/vox_hash.hpp"
#include "maphub.hpp"

enum class SearchType
{
    Distribution,
    Point,
};

template <typename T>
struct SearchRunner
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using BlockTrackType = tbb::concurrent_hash_map<Eigen::Vector3i, bool, VoxelHash>;
    using ExplorePriorityType = std::priority_queue<typename Block<T>::SearchPair, std::vector<typename Block<T>::SearchPair>, std::greater<>>;

    SearchRunner(const Eigen::Matrix<T, 3, 1> &query_point, T vox_size, size_t num_nearest, T &max_range, SearchType typ = SearchType::Distribution);

    void mark(const Eigen::Vector3i &vox);

    bool not_visited(const Eigen::Vector3i &vox) const;

    void update_search_bucket(BlockPtr<T> &block, T sq_dist = -1.0);

    void enqueue_next(ExplorePriorityType &next_block, BlockPtr<T> &curr);

    bool enough_points();

private:
    void enqueue_next_distributive(ExplorePriorityType &next_block, BlockPtr<T> &curr);

    void enqueue_next_point(ExplorePriorityType &next_block, BlockPtr<T> &curr);

    void handle_distributive(BlockPtr<T> &block, T sq_dist = -1.0);

    void handle_point(BlockPtr<T> &block);

public:
    Eigen::Vector3i node_rep;
    Eigen::Matrix<T, 3, 1> node_rep_d, qp;
    T voxel_size;

    size_t num_nearest;
    T max_range, max_range_sq;
    SearchType typ;
    BlockTrackType filter;

    // Returned options
    typename Block<T>::SearchHeap blocks;
    SearchHeap<T> points;
};

template <typename T>
using SearchRunnerVector = tbb::concurrent_vector<SearchRunner<T>>;

template <typename T>
struct Searcher
{
    // supports distribution search and point search
    using Ptr = std::shared_ptr<Searcher<T>>;

    void set_support_info(ConfigPtr<T> &config_ptr);

    void explore_tree(SearchRunner<T> &opt);

    static BlockPtrVec<T> get_matched_blocks(SearchRunner<T> &result, bool verbose);

    static AVector3TVec<T> get_matched_points(SearchRunner<T> &result, bool verbose);

    static std::vector<BlockPointPair<T>> get_matched_points_with_block(SearchRunner<T> &result, ConfigPtr<T> &config, bool verbose);

private:
    ConfigPtr<T> config = nullptr;
};

template <typename T>
using SearcherPtr = typename Searcher<T>::Ptr;
#endif
