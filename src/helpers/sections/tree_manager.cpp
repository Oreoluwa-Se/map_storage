#include "map_storage/sections/tree_manager.hpp"
#include <sstream>
#include <iomanip>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>

template <typename T>
PointStorage<T>::PointStorage(
    int max_points_in_vox, size_t max_points_in_oct_layer,
    T imbal_factor, T del_nodes_factor,
    bool track_stats, size_t init_map_size, T voxel_size)
    : config(std::make_shared<Config<T>>(max_points_in_vox, max_points_in_oct_layer, imbal_factor, del_nodes_factor, track_stats, init_map_size, voxel_size)),
      builder(std::make_shared<MapBuilder<T>>()),
      inserter(std::make_shared<Inserter<T>>()),
      searcher(std::make_shared<Searcher<T>>()),
      workpool(std::make_shared<WorkPool>()),
      deleter(std::make_shared<Deleter<T>>()),
      rebalancer(std::make_shared<Rebalancer<T>>()),
      voxel_size(voxel_size), can_use_distribution_search(track_stats) { support_functions(); }

template <typename T>
void PointStorage<T>::support_functions()
{
    config->set_support_info(workpool);
    builder->set_support_info(config);
    inserter->set_support_info(config);
    searcher->set_support_info(config);
    deleter->set_support_info(config);
    rebalancer->set_support_info(config, inserter, deleter, builder);
}

template <typename T>
bool PointStorage<T>::build(Point3dPtrVect<T> &points)
{
    return builder->build(points);
}

template <typename T>
Point3dWPtrVec<T> PointStorage<T>::get_points()
{
    size_t total_size;
    VisualizationPointStorage<T> oct_points;
    std::tie(total_size, oct_points) = config->map_points();

    Point3dWPtrVec<T> points;
    points.reserve(total_size);

    for (auto &point_sector : oct_points)
    {
        points.insert(
            points.end(),
            std::make_move_iterator(point_sector.begin()),
            std::make_move_iterator(point_sector.end()));
    }

    return points;
}

template <typename T>
void PointStorage<T>::insert(Point3dPtrVect<T> &points)
{
    BlockPtrVecCC<T> scapegoats;
    inserter->insert(points, scapegoats);

    if (scapegoats.empty())
        return;

    load_rebalance(scapegoats);
}

template <typename T>
void PointStorage<T>::print_tree()
{
    std::cout << config->get_root()->print_subtree() << std::endl;
}

template <typename T>
SearchRunner<T> PointStorage<T>::knn_search(const Eigen::Matrix<T, 3, 1> &point, size_t num_nearest, T max_range, SearchType typ)
{
    SearchRunner<T> opt(point, voxel_size, num_nearest, max_range, typ);
    if (typ == SearchType::Distribution && !can_use_distribution_search)
        return opt;

    searcher->explore_tree(opt);
    return opt;
}

template <typename T>
SearchRunner<T> PointStorage<T>::range_search(const Eigen::Matrix<T, 3, 1> &point, T max_range, SearchType typ)
{
    size_t num_nearest = std::numeric_limits<size_t>::max();
    SearchRunner<T> opt(point, voxel_size, num_nearest, max_range, typ);
    if (typ == SearchType::Distribution && !can_use_distribution_search)
        return opt;

    searcher->explore_tree(opt);
    return opt;
}

template <typename T>
SearchRunnerVector<T> PointStorage<T>::knn_search(const Point3dPtrVect<T> &points, size_t num_nearest, T max_range, SearchType typ)
{
    SearchRunnerVector<T> result;
    if (typ == SearchType::Distribution && !can_use_distribution_search)
        return result;

    result.reserve(points.size());
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, points.size()),
        [&](const tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                SearchRunner<T> opt(points[i]->point, voxel_size, num_nearest, max_range, typ);
                searcher->explore_tree(opt);
                result.emplace_back(opt);
            }
        });

    return result;
}

template <typename T>
SearchRunnerVector<T> PointStorage<T>::range_search(const Point3dPtrVect<T> &points, T max_range, SearchType typ)
{
    SearchRunnerVector<T> result;
    if (typ == SearchType::Distribution && !can_use_distribution_search)
        return result;

    result.reserve(points.size());
    size_t num_nearest = std::numeric_limits<size_t>::max();

    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, points.size()),
        [&](tbb::blocked_range<size_t> &range)
        {
            for (size_t i = range.begin(); i != range.end(); ++i)
            {
                SearchRunner<T> opt(points[i]->point, voxel_size, num_nearest, max_range, typ);
                searcher->explore_tree(opt);
                result.emplace_back(opt);
            }
        });

    return result;
}

template <typename T>
void PointStorage<T>::load_rebalance(BlockPtrVecCC<T> &scapegoats)
{
    auto voxels_ptr = std::make_shared<BlockPtrVecCC<T>>(scapegoats);
    std::shared_ptr<PointStorage<T>> workpointer = this->shared_from_this();
    workpool->enqueue_task(
        TaskType::Rebalance,
        [voxels_ptr, workpointer]()
        {
            workpointer->rebalance(*voxels_ptr);
        },
        PriorityRank::Max);
}

template <typename T>
void PointStorage<T>::rebalance(BlockPtrVecCC<T> &scapegoats)
{
    std::map<size_t, BlockPtrVec<T>> run_seq;
    rebalancer->sequencer(scapegoats, run_seq);

    // now we extract from biggest to smallest.
    for (auto it = run_seq.rbegin(); it != run_seq.rend(); ++it)
    {
        tbb::parallel_for_each(
            it->second.begin(), it->second.end(),
            [&](BlockPtr<T> &blk)
            {
                // Rebalance algorithm
                if (blk && !blk->cloned_status())
                    rebalancer->run_algo(blk);
            });
    }
}

template <typename T>
void PointStorage<T>::delete_within_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type)
{
    deleter->delete_within_points(ptd, range, del_type);
    auto blk = config->get_root();

    // check scapegoat
    if (blk->scapegoat_handler(config->imbal_factor, config->del_nodes_factor))
    {
        BlockPtrVecCC<T> scapegoats;
        scapegoats.emplace_back(blk);
        load_rebalance(scapegoats);
    }
}

template <typename T>
void PointStorage<T>::delete_outside_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type)
{
    deleter->delete_outside_points(ptd, range, del_type);
    auto blk = config->get_root();

    // check scapegoat
    if (blk->scapegoat_handler(config->imbal_factor, config->del_nodes_factor))
    {
        BlockPtrVecCC<T> scapegoats;
        scapegoats.emplace_back(blk);
        load_rebalance(scapegoats);
    }
}

template struct PointStorage<double>;
template struct PointStorage<float>;