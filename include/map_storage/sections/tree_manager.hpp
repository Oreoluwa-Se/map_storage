#ifndef TREE_MANAGER_HPP
#define TREE_MANAGER_HPP

#include <boost/thread/shared_mutex.hpp>
#include "map_storage/sections/kd_tree/maphub.hpp"
#include "map_storage/sections/worker_pool.hpp"
#include "map_storage/sections/kd_tree/builder.hpp"
#include "map_storage/sections/kd_tree/inserter.hpp"
#include "map_storage/sections/kd_tree/searcher.hpp"
#include "map_storage/sections/kd_tree/deleter.hpp"
#include "map_storage/sections/kd_tree/rebalancer.hpp"
#include "map_storage/sections/kd_tree/node.hpp"

template <typename T>
class PointStorage : public std::enable_shared_from_this<PointStorage<T>>
{
public:
    using Ptr = std::shared_ptr<PointStorage<T>>;
    PointStorage(
        int max_points_in_vox = -1, size_t max_points_in_oct_layer = 30, T imbal_factor = 0.63, T del_nodes_factor = 0.5,
        bool track_stats = false, size_t init_map_size = std::numeric_limits<size_t>::max(),
        T voxel_size = 1.0);

    void support_functions();

    bool build(Point3dPtrVect<T> &points);

    void insert(Point3dPtrVect<T> &points);

    void print_tree();

    // Seach parameters
    SearchRunner<T> knn_search(const Eigen::Matrix<T, 3, 1> &point, size_t num_nearest, T max_range, SearchType typ);

    SearchRunner<T> range_search(const Eigen::Matrix<T, 3, 1> &point, T max_range, SearchType typ);

    SearchRunnerVector<T> knn_search(const Point3dPtrVect<T> &points, size_t num_nearest, T max_range, SearchType typ);

    SearchRunnerVector<T> range_search(const Point3dPtrVect<T> &points, T max_range, SearchType typ);

    // Delete parameters
    void delete_within_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type = DeleteType::Spherical);

    void delete_outside_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type = DeleteType::Spherical);

    // point retrival
    Point3dWPtrVecCC<T> get_points();

private:
    void load_rebalance(BlockPtrVecCC<T> &scapegoats);

    void rebalance(BlockPtrVecCC<T> &scapegoats);

public:
    ConfigPtr<T> config;

private:
    MapBuilderPtr<T> builder;
    InserterPtr<T> inserter;
    SearcherPtr<T> searcher;
    DeleterPtr<T> deleter;
    RebalancerPtr<T> rebalancer;
    PoolPtr workpool;

    T voxel_size;
    bool can_use_distribution_search;
};

template <typename T>
using PointStoragePtr = typename PointStorage<T>::Ptr;
#endif