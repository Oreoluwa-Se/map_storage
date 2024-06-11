#include "map_storage/sections/kd_tree/specialized.hpp"
#include "map_storage/sections/sub_tree/base.hpp"
#include "map_storage/sections/sub_tree/octtree.hpp"
#include "map_storage/sections/tree_manager.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/bbox.hpp"
#include <Eigen/Dense>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <type_traits>
#include <utility>

// Wrapper function to measure execution time
template <typename Func, typename... Args>
typename std::enable_if<std::is_void<typename std::result_of<Func(Args...)>::type>::value>::type
measure_time(Func func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
}

// Helper function for non-void return type
template <typename Func, typename... Args>
typename std::enable_if<!std::is_void<typename std::result_of<Func(Args...)>::type>::value, typename std::result_of<Func(Args...)>::type>::type
measure_time(Func func, Args &&...args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

    return result;
}

template <typename T>
// Point3dPtrVect<T> create_random_points(size_t num_points, T min_range, T max_range, unsigned int seed = 42)
Point3dPtrVect<T> create_random_points(size_t num_points, T min_range, T max_range, unsigned int seed = std::random_device{}())
{
    Point3dPtrVect<T> points;
    points.reserve(num_points);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(min_range, max_range);

    for (size_t i = 0; i < num_points; ++i)
    {
        Eigen::Matrix<T, 3, 1> position(dis(gen), dis(gen), dis(gen));
        auto new_p = std::make_shared<Point3d<T>>(position);
        new_p->vox = Point3d<T>::calc_vox_index(new_p, 1.0);
        points.emplace_back(new_p);
    }
    // std::cout << "Point count: " << points.size() << std::endl;
    return points;
}

template <typename T>
// Point3dPtrVect<T> create_random_points(size_t num_points, T range, unsigned int seed = 987652)
Point3dPtrVect<T> create_random_points(size_t num_points, T range, unsigned int seed = std::random_device{}())
{
    Point3dPtrVect<T> points;
    points.reserve(num_points);

    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> dis(-range / 2, range / 2);

    for (size_t i = 0; i < num_points; ++i)
    {
        Eigen::Matrix<T, 3, 1> position(dis(gen), dis(gen), dis(gen));
        auto new_p = std::make_shared<Point3d<T>>(position);
        new_p->vox = Point3d<T>::calc_vox_index(new_p, 1.0);
        points.emplace_back(new_p);
    }

    // std::cout << "Point count: " << points.size() << std::endl;
    return points;
}

template <typename T>
void update(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points)
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
void update2(OctreeNodePtr<T> &node, const Point3dPtrVect<T> &points)
{
    std::cout << points.size() << std::endl;
    node->insert_points(points);
}

template <typename T>
void update4(OctreePtr<T> &node, const Point3dPtrVect<T> &points)
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
void update5(PointStoragePtr<T> &node, Point3dPtrVect<T> &points, bool verbose = false)
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
void testing_update(size_t N, T min_range, T max_range, bool track_cov)
{
    auto points = create_random_points<T>(N, max_range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "Approach 1" << std::endl;
    measure_time(update<T>, node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    points = create_random_points<T>(N, max_range);
    OctreeNodePtr<T> node_stuff_1 = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "\nApproach 2" << std::endl;
    measure_time(update2<T>, node_stuff_1, points);
    std::cout << *(node_stuff_1->bbox) << std::endl;

    points = create_random_points<T>(N, max_range);
    OctreePtr<T> node_stuff_3 = std::make_shared<Octree<T>>(30, track_cov);
    std::cout << "\nApproach 4" << std::endl;
    measure_time(update4<T>, node_stuff_3, points);
    std::cout << *(node_stuff_3->bbox) << std::endl;
}

template <typename T>
OctreeNodePtr<T> create_and_insert_node(size_t N, T min_range, T max_range, bool track_cov)
{
    auto points = create_random_points<T>(N, min_range, max_range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "Inserting Node" << std::endl;
    measure_time(update2<T>, node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;
    return node_stuff;
}

template <typename T>
OctreeNodePtr<T> create_and_insert_node(size_t N, T range, bool track_cov)
{
    auto points = create_random_points<T>(N, range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    std::cout << "Inserting Node" << std::endl;
    measure_time(update2<T>, node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    Point3dWPtrVecCC<T> pts;
    return node_stuff;
}

template <typename T>
OctreePtr<T> create_and_insert_tree(size_t N, T range, bool track_cov)
{
    auto points = create_random_points<T>(N, range);
    OctreePtr<T> node_stuff = std::make_shared<Octree<T>>(30, track_cov);

    std::cout << "Inserting tree" << std::endl;
    measure_time(update4<T>, node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    return node_stuff;
}

template <typename T>
void testing_update_del(size_t N, T min_range, T max_range, bool track_cov, bool outside)
{
    auto points = create_random_points<T>(N, min_range, max_range);
    OctreeNodePtr<T> node_stuff = std::make_shared<OctreeNode<T>>(30, track_cov);
    update<T>(node_stuff, points);
    std::cout << *(node_stuff->bbox) << std::endl;

    Eigen::Matrix<T, 3, 1> center = Eigen::Matrix<T, 3, 1>::Zero();
    T range = 0.5 * max_range;

    if (outside)
    {
        {
            auto node = create_and_insert_node(N, min_range, max_range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->outside_range_delete(center, range, DeleteType::Spherical);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;

            std::cout << *(node->bbox) << std::endl;
        }

        {
            auto node = create_and_insert_node(N, min_range, max_range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->outside_range_delete(center, range, DeleteType::Box);
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
            auto node = create_and_insert_node(N, min_range, max_range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->within_range_delete(center, range, DeleteType::Spherical);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;

            std::cout << *(node->bbox) << std::endl;
        }

        {
            auto node = create_and_insert_node(N, min_range, max_range, track_cov);
            std::cout << "\n============================" << std::endl;
            std::cout << "\nDelete within: " << range << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            auto dm = node->within_range_delete(center, range, DeleteType::Box);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
            std::cout << "Points to delete: " << dm.ptd.size() << std::endl;
            std::cout << *(node->bbox) << std::endl;
        }
    }
}

template <typename T>
void testing_search(size_t N, T min_range, T max_range, bool track_cov, size_t k, bool verbose = false)
{
    auto node = create_and_insert_node(N, max_range, track_cov);
    auto tree = create_and_insert_tree(N, max_range, track_cov);
    std::cout << "\n============================" << std::endl;

    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dis(-max_range / 2.0, max_range / 2.0);

    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
    std::cout << "\nNode Trials" << std::endl;
    {
        T range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        node->radius_search(result, qp, range, k);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    {
        T range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        node->range_search(result, qp, range);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    std::cout << "\nTree Trials" << std::endl;
    {
        T range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        tree->radius_search(result, qp, range, k);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    {
        T range = 0.3;
        SearchHeap<T> result;
        auto start = std::chrono::high_resolution_clock::now();
        tree->range_search(result, qp, range);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";

        auto res = OctreeNode<T>::get_matched(result, qp, verbose);
        std::cout << "Final size: " << res.size() << std::endl;
    }

    auto points = create_random_points<T>(2000, max_range);
    measure_time(update4<T>, tree, points);
    std::cout << *(tree->bbox) << std::endl;
}

template <typename T>
PointStoragePtr<T> test_build(size_t N, T range, bool verbose = false)
{
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>();
    // OctreeNodePtr<T> node_n = std::make_shared<OctreeNode<T>>(30, false);
    OctreeNodePtr<T> node_n = std::make_shared<OctreeNode<T>>(30, true);
    auto points = create_random_points<T>(N, range);

    measure_time(update2<T>, node_n, points);
    update5<T>(node, points, false);
    // node->print_tree();
    points = create_random_points<T>(200, range);

    // {
    //     std::cout << "\nTesting insert Octtree" << std::endl;
    //     auto start = std::chrono::high_resolution_clock::now();
    //     // std::cout << "Post pre-build" << std::endl;
    //     node_n->insert_points(points);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<T> duration = end - start;
    //     std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    // }

    {
        std::cout << "\nTesting insert KD-OCt" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        // std::cout << "Post pre-build" << std::endl;
        node->insert(points);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> duration = end - start;
        std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    }

    // node->print_tree();

    // std::cout << "Waiting till rebalance complete" << std::endl;
    // node->print_tree();
    // // if (verbose)
    // //     node->print_tree();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);
    size_t num_n = 5;
    // T s_range = range;
    T s_range = 5.0;
    // T s_range = 1.0;
    // Eigen::Matrix<T, 3, 1> qp(-3.6589, -1.0150, -2.4423);
    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
    // // {
    // //     std::cout << " ============== " << std::endl;
    // //     std::cout << "Testing Other Knn speed" << std::endl;
    // //     auto start = std::chrono::high_resolution_clock::now();
    // //     SearchHeap<T> result;
    // //     node_n->radius_search(result, qp, s_range, num_n);
    // //     auto end = std::chrono::high_resolution_clock::now();
    // //     std::chrono::duration<T> duration = end - start;
    // //     std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    // //     auto res = OctreeNode<T>::get_matched(result, qp, true);
    // //     std::cout << "Final size: " << res.size() << std::endl;
    // // }

    // {
    //     std::cout << "\nTesting Knn-Oct Search" << std::endl;
    //     auto start = std::chrono::high_resolution_clock::now();
    //     auto res = node->knn_search(qp, num_n, s_range, SearchType::Point);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<T> duration = end - start;
    //     std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    //     auto res_2 = Searcher<T>::get_matched_points(res, true);
    //     std::cout << "Final size: " << res_2.size() << std::endl;
    // }

    for (size_t idx = 0; idx < 10; ++idx)
    {
        points = create_random_points<T>(200, range);
        {
            std::cout << "\nTesting insert KD-OCt " << idx + 1 << std::endl;
            auto start = std::chrono::high_resolution_clock::now();
            // std::cout << "Post pre-build" << std::endl;
            node->insert(points);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;
            std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
        }
    }

    // // {
    // //     std::cout << "\nTesting Knn-Oct Distribution Search" << std::endl;
    // //     auto start = std::chrono::high_resolution_clock::now();
    // //     // auto res = node->knn_search(qp, num_n, s_range, SearchType::Distribution);
    // //     auto res = node->range_search(qp, s_range, SearchType::Point);
    // //     auto end = std::chrono::high_resolution_clock::now();
    // //     std::chrono::duration<T> duration = end - start;
    // //     std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    // //     // auto res_2 = Searcher<T>::get_matched_points_with_block(res, node->config, true);
    // //     auto res_2 = Searcher<T>::get_matched_points(res, true);
    // //     std::cout << "Final size: " << res_2.size() << std::endl;
    // // }
    // points = create_random_points<T>(20, range);
    // {
    //     std::cout << "\nTesting Multiple Knn-Oct Search" << std::endl;
    //     auto start = std::chrono::high_resolution_clock::now();
    //     auto res = node->knn_search(points, num_n, s_range, SearchType::Point);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<T> duration = end - start;
    //     std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    //     // auto res_2 = Searcher<T>::get_matched_points(res, true);
    //     std::cout << "Final size: " << res.size() << std::endl;
    // }

    // {
    //     std::cout << "\nTesting Knn-Oct Search 2" << std::endl;
    //     auto start = std::chrono::high_resolution_clock::now();
    //     auto res = node->knn_search(qp, num_n, s_range, SearchType::Point);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<T> duration = end - start;
    //     std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    //     auto res_2 = Searcher<T>::get_matched_points(res, true);
    //     std::cout << "Final size: " << res_2.size() << std::endl;
    // }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    node->print_tree();
    return node;
}

template <typename T>
void testing_downsample(size_t N, T range, T dwnsample_ratio = 0.5)
{
    auto points = create_random_points<T>(N, range);
    std::cout << "\nTesting Downsampling strategy" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto res = Downsample<T>::regular(points, dwnsample_ratio);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    std::cout << "Final size: " << res.size() << std::endl;
}

template <typename T>
void faster_lio_trial(size_t N, T range, size_t num_iter = 20)
{
    T tot_insert_dur = 0;
    T tot_knn_dur = 0;

    for (size_t iter = 0; iter < num_iter; ++iter)
    {
        PointStoragePtr<T> node = std::make_shared<PointStorage<T>>();
        auto points = create_random_points<T>(N, range);
        update5<T>(node, points, false);

        T insert_dur, knn_dur;

        {
            points = create_random_points<T>(200, range);
            auto start = std::chrono::high_resolution_clock::now();
            node->insert(points);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> duration = end - start;
            insert_dur = 1000 * duration.count();
        }

        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);
            size_t num_n = 5;
            Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
            T s_range = range;
            points = create_random_points<T>(5, range);
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
void testing_combined_delete(size_t N, T range)
{
    PointStoragePtr<T> node = std::make_shared<PointStorage<T>>();
    auto points = create_random_points<T>(N, range);
    update5<T>(node, points, false);
    node->print_tree();

    // deleting stuff
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(-range / 2.0, range / 2.0);

    T s_range = 1.0;
    Eigen::Matrix<T, 3, 1> qp(dis(gen), dis(gen), dis(gen));
    std::cout << "\nTesting Delete Strategy" << std::endl;
    std::cout << "Point we are deleting: " << qp.transpose() << std::endl;
    std::cout << "Range: " << s_range << std::endl;
    std::cout << " " << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    node->delete_within_points(qp, s_range, DeleteType::Spherical);
    // node->delete_outside_points(qp, s_range, DeleteType::Spherical);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<T> duration = end - start;
    std::cout << "===================================" << std::endl;
    std::cout << "Execution time: " << 1000 * duration.count() << " milliseconds\n";
    node->print_tree();
}

int main(int argc, char **argv)
{
    // size_t N = 500;
    // size_t N = 1000;
    // size_t N = 2000;
    double min_range = -5.0;
    double max_range = 5.0;

    // testing_update<double>(N, min_range, max_range, true);
    // testing_update<double>(N, min_range, max_range, false);
    // testing_update_del(N, min_range, max_range, false, false);
    // testing_update_del(N, min_range, max_range, false, true);

    // size_t k = 5;
    // testing_search(N, min_range, max_range, false, k, true);
    // testing_update(N, min_range, max_range, false);
    // test_build<double>(N, max_range, false);
    // testing_combined_delete<double>(N, max_range);
    // testing_downsample<double>(N, max_range);
    for (size_t N : {500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000, 200000, 500000})
        faster_lio_trial<double>(N, max_range, 100);

    return 0;
}