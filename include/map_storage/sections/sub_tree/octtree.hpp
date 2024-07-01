#ifndef OCTTREE_HPP
#define OCTTREE_HPP

#include "map_storage/sections/sub_tree/base.hpp"
#include "map_storage/utils/alias.hpp"
#include <condition_variable>
#include <tbb/concurrent_vector.h>
#include <boost/thread/shared_mutex.hpp>
#include <atomic>

template <typename T>
class Octree
{
public:
    using Ptr = std::shared_ptr<Octree<T>>;

    explicit Octree(size_t max_points = 30, bool track_stats = false);

    Octree(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, size_t max_points = 30, bool track_stats = false);

    void split_insert_point(const Point3dPtr<T> &point);

    void split_batch_insert();

    void radius_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k);

    void range_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range);

    void outside_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type = DeleteType::Spherical);

    void within_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type = DeleteType::Spherical);

    void gather_points(Point3dWPtrVecCC<T> &points);

    void clear();

public:
    BBoxPtr<T> bbox = nullptr;
    std::atomic<size_t> alt_size{0};

private:
    void delete_operation(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type);

private:
    size_t max_points;
    bool is_inserting;
    bool track_stats;
    std::array<OctreeNodePtr<T>, 8> roots = {nullptr};
    std::array<bool, 8> inserting_t = {false};
    std::array<boost::shared_mutex, 8> mutexes;
    Point3dPtrVectCC<T> to_flush;
    std::array<Point3dPtrVectCC<T>, 8> to_flushs;
};

template <typename T>
using OctreePtr = typename Octree<T>::Ptr;
#endif