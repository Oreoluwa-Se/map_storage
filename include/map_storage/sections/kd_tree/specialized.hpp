#ifndef SPECIALIZED_HPP
#define SPECIALIZED_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/vox_hash.hpp"
#include <boost/thread/shared_mutex.hpp>
#include <map>
#include <tbb/concurrent_unordered_set.h>
#include <set>
#include <unordered_map>
#include <utility>

namespace
{
    template <typename T>
    Eigen::Matrix<T, 1, Eigen::Dynamic> vox_weight_generator(Eigen::Matrix<T, 1, Eigen::Dynamic> &variances, Eigen::Matrix<T, 1, Eigen::Dynamic> &counts, T size)
    {
        auto round_to_nearest = [](T value)
        { return std::round(value); };

        Eigen::Matrix<T, 1, Eigen::Dynamic> weight;
        if (variances.sum() <= 0)
            weight = Eigen::Matrix<T, 1, Eigen::Dynamic>::Ones(1, variances.size()) * (1 / static_cast<T>(variances.size()));
        else
        {
            variances /= variances.sum();
            counts /= counts.sum();

            weight = 0.5 * (variances + counts);
            weight /= weight.sum();
        }

        weight *= size;
        return weight.unaryExpr(round_to_nearest);
    }
}

template <typename T>
struct VoxelData
{
    VoxelData() = default;
    void add_info(const Point3dPtr<T> &val);

    void unsafe_add_info(const Point3dPtr<T> &val);

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

    void unsafe_add_point(const Point3dPtr<T> &point);

    Point3dPtrVectCC<T> reduced(T count);

    std::pair<T, T> total_info();

    VoxelHMap points;
    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    boost::shared_mutex mtx, stats_mtx;
};

template <typename T>
struct Downsample
{

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