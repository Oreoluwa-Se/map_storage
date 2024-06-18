#ifndef SPECIALIZED_HPP
#define SPECIALIZED_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/vox_hash.hpp"
#include <boost/thread/shared_mutex.hpp>
#include <map>
#include <tbb/concurrent_unordered_set.h>
#include <unordered_map>
#include <utility>

template <typename T>
struct VoxelData
{
    VoxelData() = default;
    void add_info(const Point3dPtr<T> &val);
    void sampled_points(Point3dPtrVectCC<T> &sampled, T count);

    tbb::concurrent_vector<DistancePointPair<T>> points;
    RunningStats<T> stats;
    boost::shared_mutex mtx;
};

template <typename T>
struct DownsampleData
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using VoxelHMap = std::unordered_map<Eigen::Vector3i, VoxelData<T>, VoxelHash, VoxelHashEqual>;

    void add_point(const Point3dPtr<T> &point);

    Point3dPtrVectCC<T> reduced(T count);

    std::pair<T, T> total_info();

    static Eigen::Matrix<T, 1, Eigen::Dynamic> weight_generator(Eigen::Matrix<T, 1, Eigen::Dynamic> &variances, Eigen::Matrix<T, 1, Eigen::Dynamic> &counts, T size);

    VoxelHMap points;
    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    boost::shared_mutex mtx, stats_mtx;
};

template <typename T>
struct Downsample
{
    void regular_insert(Point3dPtr<T> &point);

    void clustered_insert(Point3dPtr<T> &point);

    std::map<T, Point3dPtrVectCC<T>> process_clustered(size_t points_size, T dwnsample_ratio);

    Point3dPtrVectCC<T> process_regular(size_t points_size, T dwnsample_ratio);

    static Eigen::Matrix<T, 1, Eigen::Dynamic> generate_weight(size_t size, T dwnsample_ratio, std::array<DownsampleData<T>, 8> &o_points);

    // Using variance based approach means we only downsample when possible. Avoids loosing key information
    template <typename PointContainer>
    static Point3dPtrVectCC<T> regular(PointContainer &vector, T dwnsample_ratio = 0.5);

    template <typename PointContainer>
    static std::map<T, Point3dPtrVectCC<T>> clustered_run(PointContainer &vector, T dwnsample_ratio = 0.5);

    static void collect_var_counts(Eigen::Matrix<T, 1, Eigen::Dynamic> &var, Eigen::Matrix<T, 1, Eigen::Dynamic> &counts, std::array<DownsampleData<T>, 8> &to_flush);

private:
    std::array<DownsampleData<T>, 8> to_flushs;
    tbb::concurrent_unordered_set<T> ts;
    std::map<T, Point3dPtrVectCC<T>> output;
    boost::shared_mutex output_mtx;
};
#endif