#ifndef BASE_STORAGE_HPP
#define BASE_STORAGE_HPP

#include "map_storage/sections/help_struct.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/bbox.hpp"
#include <boost/thread/shared_mutex.hpp>
#include <array>
#include <memory>

template <typename T>
class OctreeNode : public std::enable_shared_from_this<OctreeNode<T>>
{
public:
    using Ptr = std::shared_ptr<OctreeNode<T>>;

    using SearchPair = std::pair<T, Ptr>;

    explicit OctreeNode(size_t max_points = 30, bool track_stats = false);

    OctreeNode(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, size_t max_points = 30, bool track_stats = false);

    int get_octant(const Point3dPtr<T> &point) const;

    void insert_point(const Point3dPtr<T> &point);

    void insert_points(const Point3dPtrVect<T> &point);

    void insert_points(const Point3dPtrVectCC<T> &point);

    bool check_leaf();

    DeleteManager<T> outside_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type = DeleteType::Spherical);

    DeleteManager<T> within_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type = DeleteType::Spherical);

    static AVector3TVec<T> get_matched(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, bool verbose = false);

    void radius_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k);

    void range_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range);

    void gather_points(Point3dWPtrVecCC<T> &points);

private:
    void subdivide();

    void create_octant(const Eigen::Matrix<T, 3, 1> &center, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, int octant);

    size_t get_max_octant_count() const;

    void unsafe_insert_point(const Point3dPtr<T> &point);

    void include_point(const Point3dPtr<T> &point, int octant);

    bool empty_children();

    void handle_delete(DeleteManager<T> &to_del);

    void range_delete(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type = DeleteType::Spherical);

    void delete_aggregation(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type);

    void search_algo(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k);

public:
    BBoxPtr<T> bbox = nullptr; // bounding box
    std::array<Ptr, 8> children = {nullptr};
    mutable boost::shared_mutex mutex;

private:
    std::array<size_t, 8> octant_counts = {0};
    Point3dPtrVect<T> points;
    size_t max_points = 30;
    bool is_leaf;
};

template <typename T>
using OctreeNodePtr = typename OctreeNode<T>::Ptr;
#endif