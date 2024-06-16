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
#include <string>

template <typename T>
struct TestParams
{
    // Build parameters
    int max_points_in_vox = -1;
    size_t max_points_in_oct_layer = 30; // maximum number of points stored in each level for an octant
    T imbal_factor = 0.7;                // size discrepancy between left and right tree to trigger rebalancing
    T del_nodes_factor = 0.5;            // maximum ratio between valid and invalid nodes before rebalancing
    bool track_stats = false;            // To track the mean and covariance for each voxel - mean is used during search and covariance can be used as a validity check
    size_t init_map_size = 1;            // minimum number of points to build initial map
    T voxel_size = 1.0;                  // total voxel_size

    // Test Params
    T points_gen_range = 5.0; // cubiod
    T delete_radius = 1.0;
    T search_radius = 5.0;
    size_t num_nearest = 5;
    size_t build_size = 1000;
    size_t num_incremental_insert = 200;
    bool verbose = false;
    bool delete_within = true;
    size_t iterations_faster_lio_test = 100;
    T downsample_ratio = 0.5;
};

// Creating the functions
template <typename T>
struct RunFunctions
{

    explicit RunFunctions(bool use_config = false);

    Point3dPtrVect<T> create_random_points(size_t num_points, T range);

    void single_node_update_run(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points);

    void single_node_update_run_parallel(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points);

    void octree_ptr_run(OctreePtr<T> &node, const Point3dPtrVect<T> &points);

    void point_storage_run(PointStoragePtr<T> &node, Point3dPtrVect<T> &points);

    void testing_insert_schemes();

    OctreeNodePtr<T> create_and_insert_node();

    OctreePtr<T> create_and_insert_tree();

    void testing_octree_delete();

    void testing_search();

    PointStoragePtr<T> test_build_incremental_insert_point_storage();

    void testing_downsample_scheme();

    void faster_lio_trial(size_t build_num);

    void testing_combined_delete();

    void incremental_info();

    void test_point_retrival();

private:
    T voxel_size;
    std::mt19937 gen;
    TestParams<T> tp;
};

#endif