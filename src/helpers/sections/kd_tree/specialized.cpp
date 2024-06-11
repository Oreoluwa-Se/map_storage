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
    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = DownsampleData<T>::weight_generator(variances, counts, count);
    Point3dPtrVectCC<T> ret_points;
    int idx = 0;
    for (auto &pair : points)
    {
        pair.second.sampled_points(ret_points, weight(idx));
        ++idx;
    }

    return ret_points;
}

template <typename T>
Eigen::Matrix<T, 1, Eigen::Dynamic> DownsampleData<T>::weight_generator(Eigen::Matrix<T, 1, Eigen::Dynamic> &variances, Eigen::Matrix<T, 1, Eigen::Dynamic> &counts, T size)
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

template struct DownsampleData<double>;
template struct DownsampleData<float>;

template <typename T>
Point3dPtrVectCC<T> Downsample<T>::regular(Point3dPtrVectCC<T> &vector, T dwnsample_ratio)
{
    std::array<DownsampleData<T>, 8> to_flushs;

    // group into cardinality and variance
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, vector.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                int signc = Point3d<T>::sign_cardinality(vector[i]->point);
                to_flushs[signc].add_point(vector[i]);
            }
        });

    // get spread and count for all voxels
    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    Downsample<T>::collect_var_counts(variances, counts, to_flushs);

    // scale by maximum[we use information about spread and density]
    T d_size = dwnsample_ratio * static_cast<T>(vector.size());
    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = DownsampleData<T>::weight_generator(variances, counts, d_size);
    Point3dPtrVectCC<T> collection;
    boost::shared_mutex mtx;

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
Point3dPtrVectCC<T> Downsample<T>::regular(Point3dPtrVect<T> &vector, T dwnsample_ratio)
{
    std::array<DownsampleData<T>, 8> to_flushs;

    // group into cardinality and variance
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, vector.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                int signc = Point3d<T>::sign_cardinality(vector[i]->point);
                to_flushs[signc].add_point(vector[i]);
            }
        });

    // get spread and count for all voxels
    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    Downsample<T>::collect_var_counts(variances, counts, to_flushs);

    // scale by maximum[we use information about spread and density]
    T d_size = dwnsample_ratio * static_cast<T>(vector.size());
    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = DownsampleData<T>::weight_generator(variances, counts, d_size);
    Point3dPtrVectCC<T> collection;
    boost::shared_mutex mtx;

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
std::map<T, Point3dPtrVectCC<T>> Downsample<T>::clustered_run(Point3dPtrVectCC<T> &vector, T dwnsample_ratio)
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
                int signc = Point3d<T>::sign_cardinality(vector[i]->point);
                to_flushs[signc].add_point(vector[i]);
                if (ts.insert(vector[i]->timestamp).second)
                {
                    boost::unique_lock<boost::shared_mutex> lock(output_mtx);
                    output[vector[i]->timestamp] = Point3dPtrVectCC<T>();
                }
            }
        });

    // get spread and count for all voxels
    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    Downsample<T>::collect_var_counts(variances, counts, to_flushs);

    // scale by maximum [we use information about spread and density]
    T d_size = dwnsample_ratio * static_cast<T>(vector.size());
    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = DownsampleData<T>::weight_generator(variances, counts, d_size);

    std::vector<Point3dPtrVectCC<T>> collection(ts.size());
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
std::map<T, Point3dPtrVectCC<T>> Downsample<T>::clustered_run(Point3dPtrVect<T> &vector, T dwnsample_ratio)
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
                int signc = Point3d<T>::sign_cardinality(vector[i]->point);
                to_flushs[signc].add_point(vector[i]);
                if (ts.insert(vector[i]->timestamp).second)
                {
                    boost::unique_lock<boost::shared_mutex> lock(output_mtx);
                    output[vector[i]->timestamp] = Point3dPtrVectCC<T>();
                }
            }
        });

    Eigen::Matrix<T, 1, Eigen::Dynamic> variances, counts;
    Downsample<T>::collect_var_counts(variances, counts, to_flushs);

    // scale by maximum [we use information about spread and density]
    T d_size = dwnsample_ratio * static_cast<T>(vector.size());
    Eigen::Matrix<T, 1, Eigen::Dynamic> weight = DownsampleData<T>::weight_generator(variances, counts, d_size);

    std::vector<Point3dPtrVectCC<T>> collection(ts.size());
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