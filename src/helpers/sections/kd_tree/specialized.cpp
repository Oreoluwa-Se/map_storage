#include "map_storage/sections/kd_tree/specialized.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <algorithm>
#include <sstream>
#include <tuple>

template <typename T>
void VoxelData<T>::add_info(const Point3dPtr<T> &val)
{
    points.emplace_back(val->point.squaredNorm(), val);
    boost::unique_lock<boost::shared_mutex> lock(mtx);
    stats.add_info(val->point);
}

template <typename T>
void VoxelData<T>::unsafe_add_info(const Point3dPtr<T> &val)
{
    points.emplace_back(val->point.squaredNorm(), val);
    stats.add_info(val->point);
}

template <typename T>
void VoxelData<T>::sampled_points(Point3dPtrVectCC<T> &sampled, T count)
{
    tbb::parallel_sort(
        points.begin(), points.end(),
        [](const DistancePointPair<T> &a, const DistancePointPair<T> &b)
        {
            return a.first < b.first;
        });

    // Determine the number of points to sample
    size_t sample_count = std::max<size_t>(1, static_cast<size_t>(count));
    size_t step = points.size() / sample_count;
    if (step == 0)
        step = 1;

    // push correct point into sampled
    for (size_t idx = 0; idx < sample_count && idx * step < points.size(); ++idx)
        sampled.push_back(points[idx * step].second);
}

template struct VoxelData<double>;
template struct VoxelData<float>;

template <typename T>
void DownsampleData<T>::add_point(const Point3dPtr<T> &point)
{
    // add to the pointer list
    {
        boost::shared_lock<boost::shared_mutex> lock(mtx);
        auto key = points.find(point->vox);
        if (key != points.end())
        {
            key->second.add_info(point);
            return;
        }
    }

    boost::unique_lock<boost::shared_mutex> lock(mtx);
    auto &voxel_data = points[point->vox];
    lock.unlock();
    voxel_data.add_info(point);
}

template <typename T>
void DownsampleData<T>::unsafe_add_point(const Point3dPtr<T> &point)
{
    auto &voxel_data = points[point->vox];
    voxel_data.unsafe_add_info(point);
}

template <typename T>
std::pair<T, T> DownsampleData<T>::total_info()
{
    int map_size = points.size();
    variances = Eigen::Matrix<T, 1, Eigen::Dynamic>::Zero(1, map_size);
    counts = Eigen::Matrix<T, 1, Eigen::Dynamic>::Zero(1, map_size);
    int idx = 0;
    for (auto &pair : points)
    {
        variances(idx) = pair.second.stats.max_variance_val();
        counts(idx) = pair.second.stats.count;
        ++idx;
    }

    return {variances.sum(), counts.sum()};
}

template <typename T>
Point3dPtrVectCC<T> DownsampleData<T>::reduced(T count)
{
    // scale by maximum [we use information about spread and density]
    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = vox_weight_generator(variances, counts, count);
    Point3dPtrVectCC<T> ret_points;
    int idx = 0;
    for (auto &pair : points)
    {
        pair.second.sampled_points(ret_points, weight(idx));
        ++idx;
    }

    return ret_points;
}

template struct DownsampleData<double>;
template struct DownsampleData<float>;

template <typename T>
void Downsample<T>::collect_var_counts(Eigen::Matrix<T, 1, Eigen::Dynamic> &var, Eigen::Matrix<T, 1, Eigen::Dynamic> &counts, std::array<DownsampleData<T>, 8> &to_flushs)
{
    var = Eigen::Matrix<T, 1, 8>::Zero();
    counts = Eigen::Matrix<T, 1, 8>::Zero();
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, to_flushs.size()),
        [&](tbb::blocked_range<size_t> &range)
        {
            for (size_t idx = range.begin(); idx != range.end(); ++idx)
            {
                T variance, count;
                std::tie(variance, count) = to_flushs[idx].total_info();
                var(idx) = variance;
                counts(idx) = count;
            }
        });
}

template <typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> Downsample<T>::generate_weight(
    size_t vector_size, T dwnsample_ratio, std::array<DownsampleData<T>, 8> &o_points)
{
    // get spread and count for all voxels
    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    Downsample<T>::collect_var_counts(variances, counts, o_points);

    // scale by maximum[we use information about spread and density]
    T d_size = dwnsample_ratio * static_cast<T>(vector_size);
    return vox_weight_generator(variances, counts, d_size);
}

template <typename T>
template <typename PointContainer>
Point3dPtrVectCC<T> Downsample<T>::regular(PointContainer &vector, T dwnsample_ratio)

{
    std::array<DownsampleData<T>, 8> to_flushs;

    // group into cardinality and variance
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, vector.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
                to_flushs[vector[i]->octant_key].add_point(vector[i]);
        });

    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = Downsample<T>::generate_weight(vector.size(), dwnsample_ratio, to_flushs);
    Point3dPtrVectCC<T> collection;

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, to_flushs.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                Point3dPtrVectCC<T> res = to_flushs[i].reduced(std::max(T(1), weight(i)));
                for (auto &p : res)
                    collection.push_back(p);
            }
        });

    return collection;
}

template <typename T>
template <typename PointContainer>
std::map<T, Point3dPtrVectCC<T>> Downsample<T>::clustered_run(PointContainer &vector, T dwnsample_ratio)
{
    std::array<DownsampleData<T>, 8> to_flushs;
    tbb::concurrent_unordered_set<T> ts;
    std::map<T, Point3dPtrVectCC<T>> output;
    boost::shared_mutex output_mtx;

    // group into cardinality and variance
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, vector.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                to_flushs[vector[i]->octant_key].add_point(vector[i]);
                if (ts.insert(vector[i]->timestamp).second)
                {
                    boost::unique_lock<boost::shared_mutex> lock(output_mtx);
                    output[vector[i]->timestamp] = Point3dPtrVectCC<T>();
                }
            }
        });

    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = Downsample<T>::generate_weight(vector.size(), dwnsample_ratio, to_flushs);

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, to_flushs.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                auto res = to_flushs[i].reduced(std::max(T(1.0), weight(i)));
                for (auto &point : res)
                    output[point->timestamp].push_back(point);
            }
        });

    return output;
}

template struct Downsample<double>;
template struct Downsample<float>;

template Point3dPtrVectCC<double> Downsample<double>::regular<Point3dPtrVectCC<double>>(Point3dPtrVectCC<double> &, double);
template Point3dPtrVectCC<float> Downsample<float>::regular<Point3dPtrVectCC<float>>(Point3dPtrVectCC<float> &, float);

template Point3dPtrVectCC<double> Downsample<double>::regular<Point3dPtrVect<double>>(Point3dPtrVect<double> &, double);
template Point3dPtrVectCC<float> Downsample<float>::regular<Point3dPtrVect<float>>(Point3dPtrVect<float> &, float);

template std::map<double, Point3dPtrVectCC<double>> Downsample<double>::clustered_run<Point3dPtrVectCC<double>>(Point3dPtrVectCC<double> &, double);
template std::map<float, Point3dPtrVectCC<float>> Downsample<float>::clustered_run<Point3dPtrVectCC<float>>(Point3dPtrVectCC<float> &, float);

template std::map<double, Point3dPtrVectCC<double>> Downsample<double>::clustered_run<Point3dPtrVect<double>>(Point3dPtrVect<double> &, double);
template std::map<float, Point3dPtrVectCC<float>> Downsample<float>::clustered_run<Point3dPtrVect<float>>(Point3dPtrVect<float> &, float);
