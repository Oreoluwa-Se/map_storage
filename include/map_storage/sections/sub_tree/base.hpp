#ifndef BASE_STORAGE_HPP
#define BASE_STORAGE_HPP

#include "map_storage/sections/help_struct.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/bbox.hpp"
#include <boost/thread/shared_mutex.hpp>
#include <array>
#include <memory>
#include <atomic>

template <typename T>
class OctreeNode : public std::enable_shared_from_this<OctreeNode<T>>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<OctreeNode<T>>;

    using SearchPair = std::pair<T, Ptr>;

    explicit OctreeNode(size_t max_points = 30, bool track_stats = false);

    OctreeNode(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, size_t max_points = 30, bool track_stats = false);

    void insert_point(const Point3dPtr<T> &point);

    bool check_leaf();

    DeleteManager<T> outside_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type = DeleteType::Spherical);

    DeleteManager<T> within_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type = DeleteType::Spherical);

    static AVector3TVec<T> get_matched(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, bool verbose = false);

    void radius_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k);

    void range_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range);

    void gather_points(Point3dWPtrVecCC<T> &points);

private:
    void subdivide();

    bool move_left(const Eigen::Matrix<T, 3, 1> &pt);

    void create_child(const Eigen::Matrix<T, 3, 1> &center, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, int octant);

    void include_point(Point3dPtr<T> &&point);

    bool empty_children();

    void handle_delete(DeleteManager<T> &to_del);

    void range_delete(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type = DeleteType::Spherical);

    void process_leaf_node(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, T &range_sq, size_t k);

    void search_algo(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k);

    bool skippable_del_node_ops(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type);

    void top_down_update_bbox_info();

    void leaf_vector_delete(AVector3TVec<T> &ptd, const Eigen::Matrix<T, 3, 1> &center, T range_sq, DeleteCondition cond);

public:
    BBoxPtr<T> bbox = nullptr; // bounding box
    std::array<Ptr, 2> children = {nullptr};
    mutable boost::shared_mutex mutex;

private:
    Point3dPtrVectCC<T> points;
    std::atomic<int> curr_size{0};
    Eigen::Matrix<T, 3, 1> split_center;
    int depth = 0;
    size_t max_points = 30;
    bool is_leaf;
};

template <typename T>
using OctreeNodePtr = typename OctreeNode<T>::Ptr;
#endif