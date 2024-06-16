#include "run.hpp"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "map_storage/utils/timer.hpp"
#include "map_storage/utils/loader.hpp"

template <typename T>
RunFunctions<T>::RunFunctions(bool use_config)
{
    if (use_config)
    {
        set_param(tp.max_points_in_vox, build_params["max_points_in_vox"]);
        set_param(tp.max_points_in_oct_layer, build_params["max_points_in_oct_layer"]);
        set_param(tp.imbal_factor, build_params["imbal_factor"]);
        set_param(tp.del_nodes_factor, build_params["del_nodes_factor"]);
        set_param(tp.track_stats, build_params["track_stats"]);
        set_param(tp.init_map_size, build_params["init_map_size"]);
        set_param(tp.voxel_size, build_params["voxel_size"]);

        set_param(tp.points_gen_range, test_params["points_gen_range"]);
        set_param(tp.delete_radius, test_params["delete_radius"]);
        set_param(tp.delete_within, test_params["delete_within"]);
        set_param(tp.search_radius, test_params["search_radius"]);
        set_param(tp.num_nearest, test_params["num_nearest"]);
        set_param(tp.build_size, test_params["build_size"]);
        set_param(tp.num_incremental_insert, test_params["num_incremental_insert"]);
        set_param(tp.verbose, test_params["verbose"]);
        set_param(tp.iterations_faster_lio_test, test_params["iterations_faster_lio_test"]);
        set_param(tp.downsample_ratio, test_params["downsample_ratio"]);
    }
    // base parameters

    std::random_device rd;
    std::mt19937 gen_2(rd());
    gen = gen_2;
}

// Creators
template <typename T>
Point3dPtrVect<T> RunFunctions<T>::create_random_points(size_t num_points, T range)
{
    Point3dPtrVect<T> points;
    points.reserve(num_points);

    std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);

    for (size_t i = 0; i < num_points; ++i)
    {
        Eigen::Matrix<T, 3, 1> position(dis(gen), dis(gen), dis(gen));
        auto new_p = std::make_shared<Point3d<T>>(position);
        new_p->vox = Point3d<T>::calc_vox_index(new_p, tp.voxel_size);
        points.emplace_back(new_p);
    }

    if (tp.verbose)
        std::cout << "Point count: " << points.size() << std::endl;

    return points;
}

template <typename T>
void RunFunctions<T>::single_node_update_run(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points)
{
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, points.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
            for (size_t i = r.begin(); i != r.end(); ++i)
                node->insert_point(points[i]);
        });
}

template <typename T>
void RunFunctions<T>::single_node_update_run_parallel(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points)
{
    if (tp.verbose)
        std::cout << points.size() << std::endl;
    node->insert_points(points);
}

template <typename T>
void RunFunctions<T>::octree_ptr_run(OctreePtr<T> &node, const Point3dPtrVect<T> &points)
{
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, points.size()),
        [&](const tbb::blocked_range<size_t> &r)
        {
            for (size_t i = r.begin(); i != r.end(); ++i)
                node->split_insert_point(points[i]);
        });

    node->split_batch_insert();
}

template <typename T>
void RunFunctions<T>::point_storage_run(PointStoragePtr<T> &node, Point3dPtrVect<T> &points)
{
    auto start = std::chrono::high_resolution_clock::now();
    node->build(points);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> duration = end - start;

    if (tp.verbose)
    {
        std::cout << "Insertion Execution time: " << 1000 * duration.count() << " milliseconds\n";
        node->print_tree();
    }
}

template <typename T>
OctreeNodePtr<T> RunFunctions<T>::create_and_insert_node()
{
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(tp.max_points_in_vox, tp.track_stats);
    std::cout << "Inserting Node" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run_parallel, this, node_stuff, points);
    if (tp.verbose)
        std::cout << *(node_stuff->bbox) << std::endl;

    Point3dWPtrVecCC<T> pts;
    return node_stuff;
}

// Testing
template <typename T>
void RunFunctions<T>::testing_insert_schemes()
{
    std::cout << "Testing Different insertion types" << std::endl;
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(tp.max_points_in_vox, tp.track_stats);
    std::cout << "Single thread run:" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run, this, node_stuff, points);

    std::cout << *(node_stuff->bbox) << std::endl;

    points = create_random_points(tp.build_size, tp.points_gen_range);
    OctreeNodePtr<T> node_stuff_1 = std::make_shared<OctreeNode<T>>(tp.max_points_in_vox, tp.track_stats);
    std::cout << "\nParallel thread run:" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run_parallel, this, node_stuff_1, points);
    std::cout << *(node_stuff_1->bbox) << std::endl;

    points = create_random_points(tp.build_size, tp.points_gen_range);
    OctreePtr<T> node_stuff_3 = std::make_shared<Octree<T>>(tp.max_points_in_vox, tp.track_stats);
    std::cout << "\nParallel thread + Flush operation" << std::endl;
    measure_time(&RunFunctions<T>::octree_ptr_run, this, node_stuff_3, points);
    std::cout << *(node_stuff_3->bbox) << std::endl;
}

template <typename T>
OctreePtr<T> RunFunctions<T>::create_and_insert_tree()
{
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    OctreePtr<T> node_stuff = std::make_shared<Octree<T>>(tp.max_points_in_vox, tp.track_stats);

    std::cout << "Inserting tree" << std::endl;
    measure_time(&RunFunctions<T>::octree_ptr_run, this, node_stuff, points);
    if (tp.verbose)
        std::cout << *(node_stuff->bbox) << std::endl;

    return node_stuff;
}

template <typename T>
void RunFunctions<T>::testing_octree_delete()
{
    OctreeNodePtr<T> node = std::make_shared<OctreeNode<T>>(tp.max_points_in_vox, tp.track_stats);
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    single_node_update_run_parallel(node, points);
    if (tp.verbose)
        std::cout << *(node->bbox) << std::endl;

    Eigen::Matrix<T, 3, 1> center = Eigen::Matrix<T, 3, 1>::Zero();
    T s_range = tp.delete_radius;
    std::chrono::duration<double> duration;
    DeleteManager<T> dm;

    if (!tp.delete_within)
    {
        std::cout << "\n============================" << std::endl;
        std::cout << "\nDelete outside: " << s_range << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        dm = node->outside_range_delete(center, s_range, DeleteType::Spherical);
        auto end = std::chrono::high_resolution_clock::now();

        duration = end - start;
    }
    else
    {

        std::cout << "\n============================" << std::endl;
        std::cout << "\nDelete within: " << s_range << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        dm = node->within_range_delete(center, s_range, DeleteType::Box);
        auto end = std::chrono::high_resolution_clock::now();

        duration = end - start;
    }

    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    std::cout << "Points to delete: " << dm.ptd.size() << std::endl;
    if (tp.verbose)
        std::cout << *(node->bbox) << std::endl;
}

template <typename T>
void RunFunctions<T>::testing_search()
{
    std::cout << "\nTesting the search scheme" << std::endl;
    std::cout << "Note: Node and Tree use different build points." << std::endl;
    std::cout << "      The focus here is the speed gain between regular octree and the cardinal octree " << std::endl;
    bool prev_verb = tp.verbose;
    tp.verbose = false;
    auto node = create_and_insert_node();
    auto tree = create_and_insert_tree();
    std::cout << "Number of points to search from: " << tp.build_size << std::endl;
    std::cout << "============================\n";
    tp.verbose = prev_verb;
    std::uniform_real_distribution<T>
        dis(-tp.points_gen_range / 2.0, tp.points_gen_range / 2.0);
    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
    std::cout << "Basic Octree Trial" << std::endl;
    {
        std::cout << "Radius search:" << std::endl;
        T search_range = tp.search_radius;
        SearchHeap<T> result;

        auto start = std::chrono::high_resolution_clock::now();
        node->radius_search(result, qp, search_range, tp.num_nearest);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;

        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        auto res = OctreeNode<T>::get_matched(result, qp, tp.verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    {
        std::cout << "Range search:" << std::endl;
        T search_range = tp.search_radius;
        SearchHeap<T> result;

        auto start = std::chrono::high_resolution_clock::now();
        node->range_search(result, qp, search_range);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        auto res = OctreeNode<T>::get_matched(result, qp, tp.verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    std::cout << "\nCardinal Tree Trial" << std::endl;
    {
        std::cout << "Radius search:" << std::endl;
        T search_range = tp.search_radius;
        SearchHeap<T> result;

        auto start = std::chrono::high_resolution_clock::now();
        tree->radius_search(result, qp, search_range, tp.num_nearest);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;

        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        auto res = OctreeNode<T>::get_matched(result, qp, tp.verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    {
        std::cout << "Range search:" << std::endl;
        T search_range = tp.search_radius;
        SearchHeap<T> result;

        auto start = std::chrono::high_resolution_clock::now();
        tree->range_search(result, qp, search_range);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;

        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        auto res = OctreeNode<T>::get_matched(result, qp, tp.verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }
}

template <typename T>
PointStoragePtr<T> RunFunctions<T>::test_build_incremental_insert_point_storage()
{
    std::cout << "\nTesting insert KD-Variant Incrementally" << std::endl;
    size_t tot_points = tp.build_size;
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>(
        tp.max_points_in_vox, tp.max_points_in_oct_layer, tp.imbal_factor, tp.del_nodes_factor,
        tp.track_stats, tp.init_map_size, tp.voxel_size);

    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    point_storage_run(node, points);

    std::cout << "\n Inserting" << tp.num_incremental_insert << "new points" << std::endl;
    points = create_random_points(tp.num_incremental_insert, tp.points_gen_range);
    {
        tot_points += tp.num_incremental_insert;
        auto start = std::chrono::high_resolution_clock::now();
        node->insert(points);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    }

    if (tp.verbose)
        node->print_tree();

    std::cout << "\nIncremental insert" << tp.iterations_faster_lio_test << "times " << std::endl;
    T total_time = 0;
    size_t tot_inc = tp.iterations_faster_lio_test;
    for (size_t idx = 0; idx < tot_inc; ++idx)
    {
        points = create_random_points(tp.num_incremental_insert, tp.points_gen_range);
        {
            std::cout << "\nInserting stage: " << idx + 1 << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            node->insert(points);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;
            total_time += 1000 * duration.count();
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        }

        tot_points += tp.num_incremental_insert;
    }

    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "\nTotal expected points: " << tot_points << std::endl;
    std::cout << "Average insert: " << T(total_time / static_cast<T>(tot_inc)) << std::endl;
    std::cout << "=======================\n";

    if (tp.verbose)
        node->print_tree();

    return node;
}

template <typename T>
void RunFunctions<T>::testing_downsample_scheme()
{
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    std::cout << "\nTesting Downsampling strategy" << std::endl;
    std::cout << "=======================\n";

    auto start = std::chrono::high_resolution_clock::now();
    auto res = Downsample<T>::regular(points, tp.downsample_ratio);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    std::cout << "Final size: " << res.size() << std::endl;
}

template <typename T>
void RunFunctions<T>::faster_lio_trial(size_t N)
{
    T tot_insert_dur = 0;
    T tot_knn_dur = 0;

    tp.verbose = false;

    for (size_t iter = 0; iter < tp.iterations_faster_lio_test; ++iter)
    {
        PointStoragePtr<T> node = std::make_shared<PointStorage<T>>(
            tp.max_points_in_vox, tp.max_points_in_oct_layer, tp.imbal_factor, tp.del_nodes_factor,
            tp.track_stats, tp.init_map_size, tp.voxel_size);

        auto points = create_random_points(N, tp.points_gen_range);
        point_storage_run(node, points);

        T insert_dur, knn_dur;
        {
            points = create_random_points(tp.num_incremental_insert, tp.points_gen_range);

            auto start = std::chrono::high_resolution_clock::now();
            node->insert(points);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<T> duration = end - start;
            insert_dur = 1000 * duration.count();
        }

        {
            std::uniform_real_distribution<T> dis(-tp.points_gen_range / 2.0, tp.points_gen_range / 2.0);
            Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
            points = create_random_points(5, tp.points_gen_range);

            auto start = std::chrono::high_resolution_clock::now();
            T s_range = tp.points_gen_range;
            auto res = node->knn_search(points, tp.num_nearest, s_range, SearchType::Point);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;

            knn_dur = 1000 * duration.count();
        }

        node.reset();
        tot_insert_dur += insert_dur;
        tot_knn_dur += knn_dur;
    }

    T avg_insert_dur = tot_insert_dur / tp.iterations_faster_lio_test;
    T avg_knn_dur = tot_knn_dur / tp.iterations_faster_lio_test;
    std::cout << "\n============================" << std::endl;
    std::cout << "Average insert duration: " << avg_insert_dur << std::endl;
    std::cout << "Average knn duration: " << avg_knn_dur << std::endl;
}

template <typename T>
void RunFunctions<T>::test_point_retrival()
{
    std::cout << "Max point in vox: " << tp.max_points_in_vox << std::endl;
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>(
        tp.max_points_in_vox, tp.max_points_in_oct_layer, tp.imbal_factor, tp.del_nodes_factor,
        tp.track_stats, tp.init_map_size, tp.voxel_size);

    std::cout << "\nTesting Point Retrival" << std::endl;
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    point_storage_run(node, points);

    auto start = std::chrono::high_resolution_clock::now();
    Point3dWPtrVecCC<T> pts = node->get_points();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> duration = end - start;

    std::cout << "Total number of points extracted: " << pts.size() << " in " << duration.count() * 1000 << " millseconds" << std::endl;
}

template <typename T>
void RunFunctions<T>::testing_combined_delete()
{
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>(
        tp.max_points_in_vox, tp.max_points_in_oct_layer, tp.imbal_factor, tp.del_nodes_factor,
        tp.track_stats, tp.init_map_size, tp.voxel_size);

    std::cout << "\nTesting Delete Strategy" << std::endl;
    auto points = create_random_points(tp.build_size, tp.points_gen_range);
    point_storage_run(node, points);

    std::uniform_real_distribution<T> dis(-tp.points_gen_range / 2.0, tp.points_gen_range / 2.0);
    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));

    // deleting stuff

    std::cout << "Point we are deleting: " << Point3d<T>::eig_to_string(qp) << std::endl;
    std::cout << "Delete Radius: " << tp.delete_radius << std::endl;
    std::cout << "==================================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    node->delete_within_points(qp, tp.delete_radius, DeleteType::Spherical);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<T> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

    if (tp.verbose)
    {
        std::cout << "Post delete Map" << std::endl;
        node->print_tree();
    }
}

template <typename T>
void RunFunctions<T>::incremental_info()
{
    for (size_t N : {500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000, 200000, 500000})
        faster_lio_trial(N);
}

template struct RunFunctions<double>;
template struct RunFunctions<float>;