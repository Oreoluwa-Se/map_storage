#include "run.hpp"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include "map_storage/utils/timer.hpp"

template <typename T>
RunFunctions<T>::RunFunctions(T voxel_size) : voxel_size(voxel_size)
{
    std::random_device rd;
    std::mt19937 gen_2(rd());
    gen = gen_2;
}

template <typename T>
Point3dPtrVect<T> RunFunctions<T>::create_random_points(size_t num_points, T range, bool verbose)
{
    Point3dPtrVect<T> points;
    points.reserve(num_points);

    std::uniform_real_distribution<T> dis(-range / 2, range / 2);

    for (size_t i = 0; i < num_points; ++i)
    {
        Eigen::Matrix<T, 3, 1> position(dis(gen), dis(gen), dis(gen));
        auto new_p = std::make_shared<Point3d<T>>(position);
        new_p->vox = Point3d<T>::calc_vox_index(new_p, 1.0);
        points.emplace_back(new_p);
    }

    if (verbose)
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
void RunFunctions<T>::point_storage_run(PointStoragePtr<T> &node, Point3dPtrVect<T> &points, bool verbose)
{
    auto start = std::chrono::high_resolution_clock::now();
    node->build(points);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> duration = end - start;

    if (verbose)
    {
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        node->print_tree();
    }
}

template <typename T>
void RunFunctions<T>::testing_insert_schemes(size_t N, T range, bool track_cov)
{
    auto points = create_random_points(N, range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "Approach 1" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run, this, node_stuff, points);

    std::cout << *(node_stuff->bbox) << std::endl;

    points = create_random_points(N, range);
    OctreeNodePtr<T> node_stuff_1 = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "\nApproach 2" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run_parallel, this, node_stuff_1, points);
    std::cout << *(node_stuff_1->bbox) << std::endl;

    points = create_random_points(N, range);
    OctreePtr<T> node_stuff_3 = std::make_shared<Octree<T>>(30, track_cov);
    std::cout << "\nApproach 3" << std::endl;
    measure_time(&RunFunctions<T>::octree_ptr_run, this, node_stuff_3, points);
    std::cout << *(node_stuff_3->bbox) << std::endl;
}

template <typename T>
OctreeNodePtr<T> RunFunctions<T>::create_and_insert_node(size_t N, T min_range, T range, bool track_cov)
{
    auto points = create_random_points(N, min_range, range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "Inserting Node" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run_parallel, this, node_stuff, points);

    std::cout << *(node_stuff->bbox) << std::endl;
    return node_stuff;
}

template <typename T>
OctreeNodePtr<T> RunFunctions<T>::create_and_insert_node(size_t N, T range, bool track_cov)
{
    auto points = create_random_points(N, range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "Inserting Node" << std::endl;
    measure_time(&RunFunctions<T>::single_node_update_run_parallel, this, node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    Point3dWPtrVecCC<T> pts;
    return node_stuff;
}

template <typename T>
OctreePtr<T> RunFunctions<T>::create_and_insert_tree(size_t N, T range, bool track_cov)
{
    auto points = create_random_points(N, range);
    OctreePtr<T> node_stuff = std::make_shared<Octree<T>>(30, track_cov);

    std::cout << "Inserting tree" << std::endl;
    measure_time(&RunFunctions<T>::octree_ptr_run, this, node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    return node_stuff;
}

template <typename T>
void RunFunctions<T>::testing_delete_scheme(size_t N, T range, bool track_cov, bool outside)
{
    auto points = create_random_points(N, range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    single_node_update_run_parallel(node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    Eigen::Matrix<T, 3, 1> center = Eigen::Matrix<T, 3, 1>::Zero();
    T s_range = 0.5 * range;

    if (outside)
    {
        {
            auto node = create_and_insert_node(N, -range, range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->outside_range_delete(center, s_range, DeleteType::Spherical);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;

            std::cout << *(node->bbox) << std::endl;
        }

        {
            auto node = create_and_insert_node(N, -range, range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->outside_range_delete(center, s_range, DeleteType::Box);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;
            std::cout << *(node->bbox) << std::endl;
        }
    }
    else
    {
        {
            auto node = create_and_insert_node(N, -range, range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->within_range_delete(center, s_range, DeleteType::Spherical);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;

            std::cout << *(node->bbox) << std::endl;
        }

        {
            auto node = create_and_insert_node(N, -range, range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->within_range_delete(center, s_range, DeleteType::Box);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;
            std::cout << *(node->bbox) << std::endl;
        }
    }
}

template <typename T>
void RunFunctions<T>::testing_search(size_t N, T range, bool track_cov, size_t k, bool verbose)
{
    auto node = create_and_insert_node(N, range, track_cov);
    auto tree = create_and_insert_tree(N, range, track_cov);
    std::cout << "\n============================" << std::endl;

    std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);
    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
    std::cout << "\nNode Trials" << std::endl;
    {
        T search_range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        node->radius_search(result, qp, search_range, k);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    {
        T search_range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        node->range_search(result, qp, search_range);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    std::cout << "\nTree Trials" << std::endl;
    {
        T search_range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        tree->radius_search(result, qp, search_range, k);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    {
        T search_range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        tree->range_search(result, qp, search_range);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }
}

template <typename T>
PointStoragePtr<T> RunFunctions<T>::test_build_incremental_insert_point_storage(size_t N, T range, size_t num_incremental_insert, bool verbose)
{
    std::cout << "\nTesting insert KD-Variant Incrementally" << std::endl;
    size_t tot_points = range;
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>();
    auto points = create_random_points(N, range);
    point_storage_run(node, points, verbose);

    std::cout << "\n Inserting 200 new points" << std::endl;
    points = create_random_points(num_incremental_insert, range);
    {
        tot_points += num_incremental_insert;
        auto start = std::chrono::high_resolution_clock::now();
        node->insert(points);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    }

    if (verbose)
        node->print_tree();

    std::cout << "\nIncremental insert 10 times" << std::endl;
    T total_time = 0;
    size_t tot_inc = 10;
    for (size_t idx = 0; idx < tot_inc; ++idx)
    {
        points = create_random_points(num_incremental_insert, range);
        {
            std::cout << "\nInserting stage: " << idx + 1 << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            node->insert(points);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;
            total_time += 1000 * duration.count();
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        }

        tot_points += num_incremental_insert;
    }

    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "\nTotal expected points: " << tot_points << std::endl;
    std::cout << "Average insert: " << T(total_time / static_cast<T>(tot_inc)) << std::endl;
    std::cout << "=======================\n";

    if (verbose)
        node->print_tree();

    return node;
}

template <typename T>
void RunFunctions<T>::testing_downsample_scheme(size_t N, T range, T dwnsample_ratio)
{
    auto points = create_random_points(N, range);
    std::cout << "\nTesting Downsampling strategy" << std::endl;
    std::cout << "=======================";
    auto start = std::chrono::high_resolution_clock::now();
    auto res = Downsample<T>::regular(points, dwnsample_ratio);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    std::cout << "Final size: " << res.size() << std::endl;
}

template <typename T>
void RunFunctions<T>::faster_lio_trial(size_t N, T range, size_t num_iter, size_t num_insert, size_t num_search)
{
    T tot_insert_dur = 0;
    T tot_knn_dur = 0;

    for (size_t iter = 0; iter < num_iter; ++iter)
    {
        PointStoragePtr<T> node = std::make_shared<PointStorage<T>>();
        auto points = create_random_points(N, range);
        point_storage_run(node, points, false);

        T insert_dur, knn_dur;
        {
            points = create_random_points(num_insert, range);
            auto start = std::chrono::high_resolution_clock::now();
            node->insert(points);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;
            insert_dur = 1000 * duration.count();
        }

        {
            std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);
            size_t num_n = num_insert;
            Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
            T s_range = range;
            points = create_random_points(num_search, range);
            auto start = std::chrono::high_resolution_clock::now();
            auto res = node->knn_search(points, num_n, s_range, SearchType::Point);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;
            knn_dur = 1000 * duration.count();
        }

        node.reset();
        tot_insert_dur += insert_dur;
        tot_knn_dur += knn_dur;
    }

    T avg_insert_dur = tot_insert_dur / num_iter;
    T avg_knn_dur = tot_knn_dur / num_iter;
    std::cout << "\n============================" << std::endl;
    std::cout << "Average insert duration: " << avg_insert_dur << std::endl;
    std::cout << "Average knn duration: " << avg_knn_dur << std::endl;
}

template <typename T>
void RunFunctions<T>::testing_combined_delete(size_t N, T range, T delete_range, bool verbose)
{
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>();
    auto points = create_random_points(N, range);
    point_storage_run(node, points, false);
    if (verbose)
        node->print_tree();

    // deleting stuff
    std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);

    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
    std::cout << "\nTesting Delete Strategy" << std::endl;
    std::cout << "Point we are deleting: " << qp.transpose() << std::endl;
    std::cout << "Range: " << delete_range << std::endl;
    std::cout << "==================================" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    node->delete_within_points(qp, delete_range, DeleteType::Spherical);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<T> duration = end - start;
    std::cout << "===================================" << std::endl;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

    if (verbose)
        node->print_tree();
}

template <typename T>
void RunFunctions<T>::incremental_info(T range, size_t test_run)
{
    for (size_t N : {500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000, 200000, 500000})
        faster_lio_trial(N, range, test_run);
}

template struct RunFunctions<double>;
template struct RunFunctions<float>;