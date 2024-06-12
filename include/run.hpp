#ifndef RUN_HPP
#define RUN_HPP

#include "map_storage/sections/kd_tree/specialized.hpp"
#include "map_storage/sections/sub_tree/base.hpp"
#include "map_storage/sections/sub_tree/octtree.hpp"
#include "map_storage/sections/tree_manager.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/bbox.hpp"
#include <Eigen/Dense>
#include <random>

// Creating the functions
template <typename T>
struct RunFunctions
{
    RunFunctions(T voxel_size = 1.0);

    Point3dPtrVect<T> create_random_points(size_t num_points, T range, bool verbose = false);

    void single_node_update_run(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points);

    void single_node_update_run_parallel(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points);

    void octree_ptr_run(OctreePtr<T> &node, const Point3dPtrVect<T> &points);

    void point_storage_run(PointStoragePtr<T> &node, Point3dPtrVect<T> &points, bool verbose = false);

    void testing_insert_schemes(size_t N, T range, bool track_cov);

    OctreeNodePtr<T> create_and_insert_node(size_t N, T min_range, T range, bool track_cov);

    OctreeNodePtr<T> create_and_insert_node(size_t N, T range, bool track_cov);

    OctreePtr<T> create_and_insert_tree(size_t N, T range, bool track_cov);

    void testing_delete_scheme(size_t N, T range, bool track_cov, bool outside);

    void testing_search(size_t N, T range, bool track_cov, size_t k, bool verbose = false);

    PointStoragePtr<T> test_build_incremental_insert_point_storage(size_t N, T range, size_t num_incremental_insert = 200, bool verbose = false);

    void testing_downsample_scheme(size_t N, T range, T dwnsample_ratio = 0.5);

    void faster_lio_trial(size_t N, T range, size_t num_iter = 20, size_t num_insert = 200, size_t num_search = 5);

    void testing_combined_delete(size_t N, T range, T delete_range = 1.0, bool verbose = false);

    void incremental_info(T range, size_t test_run = 100);

private:
    T voxel_size;
    std::mt19937 gen;
};

#endif