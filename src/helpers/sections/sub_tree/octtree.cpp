#include "map_storage/sections/sub_tree/octtree.hpp"

template <typename T>
Octree<T>::Octree(size_t max_points, bool track_stats, int total_allowed_points)
    : bbox(std::make_shared<BBox<T>>(track_stats)),
      max_points(max_points), track_stats(track_stats), total_allowed_points(total_allowed_points) {}

template <typename T>
Octree<T>::Octree(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, size_t max_points, bool track_stats, int total_allowed_points)
    : bbox(std::make_shared<BBox<T>>(min, max, track_stats)),
      max_points(max_points), track_stats(track_stats), total_allowed_points(total_allowed_points) {}

template <typename T>
void Octree<T>::clear()
{
    // clearing out octnodes
    for (auto &node : roots)
        node.reset();
}

template <typename T>
bool Octree<T>::can_insert_new_point()
{
    if (total_allowed_points == -1.0)
        return true;

    return alt_size.load(std::memory_order_relaxed) <= total_allowed_points;
}

template <typename T>
void Octree<T>::split_insert_point(const Point3dPtr<T> &point)
{
    if (!can_insert_new_point())
        return;

    alt_size.fetch_add(1, std::memory_order_relaxed);
    bbox->update(point->point);
    {
        // if inserting just add it to list of nodes to insert
        boost::shared_lock<boost::shared_mutex> lock_s(mutexes[point->octant_key]);
        if (roots[point->octant_key])
        {
            roots[point->octant_key]->insert_point(point);
            return;
        }
    }

    {
        boost::unique_lock<boost::shared_mutex> lock(mutexes[point->octant_key]);
        if (!roots[point->octant_key]) // create coordinate
            roots[point->octant_key] = std::make_shared<OctreeNode<T>>(max_points, false);
    }
    roots[point->octant_key]->insert_point(point);
}

template <typename T>
void Octree<T>::radius_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k)
{
    if (BBox<T>::Status::Outside == bbox->point_within_bbox(qp, range))
        return;

    int sign_c = Point3d<T>::sign_cardinality(qp);
    int curr_idx = sign_c;

    do
    {
        if (roots[curr_idx])
        {
            boost::shared_lock<boost::shared_mutex> lock_s(mutexes[sign_c]);
            roots[curr_idx]->radius_search(result, qp, range, k);
        }

        // modded loop through
        curr_idx = (curr_idx + 1) % 8;
    } while (curr_idx != sign_c);
}

template <typename T>
void Octree<T>::range_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range)
{
    if (BBox<T>::Status::Outside == bbox->point_within_bbox(qp, range))
        return;

    int sign_c = Point3d<T>::sign_cardinality(qp);
    int curr_idx = sign_c;
    do
    {
        if (roots[curr_idx])
        {
            boost::shared_lock<boost::shared_mutex> lock_s(mutexes[sign_c]);
            roots[curr_idx]->range_search(result, qp, range);
        }

        // modded loop through
        curr_idx = (curr_idx + 1) % 8;
    } while (curr_idx != sign_c);
}

template <typename T>
void Octree<T>::delete_operation(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type)
{
    DeleteManager<T> to_del;
    int sign_c = Point3d<T>::sign_cardinality(center);
    int curr_idx = sign_c;
    do
    {
        if (roots[curr_idx])
        {
            boost::unique_lock<boost::shared_mutex> lock_s(mutexes[sign_c]);
            DeleteManager<T> local = roots[curr_idx]->outside_range_delete(center, range, del_type);
            to_del.update(local.min, local.max);
            std::move(local.ptd.begin(), local.ptd.end(), std::back_inserter(to_del.ptd));
        }

        // modded loop through
        curr_idx = (curr_idx + 1) % 8;
    } while (curr_idx != sign_c);

    // update final bounding box
    bbox->min_max_update(to_del.min, to_del.max);
    bbox->decrement(to_del.ptd);
}

template <typename T>
void Octree<T>::outside_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type)
{
    auto del_status = DeleteManager<T>::skip_criteria(DeleteCondition::Outside, del_type, bbox, center, range);
    if (del_status == DeleteStatus::Skip)
        return;

    delete_operation(center, range, del_type);
}

template <typename T>
void Octree<T>::within_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type)
{
    auto del_status = DeleteManager<T>::skip_criteria(DeleteCondition::Inside, del_type, bbox, center, range);
    if (del_status == DeleteStatus::Skip)
        return;

    delete_operation(center, range, del_type);
}

template <typename T>
void Octree<T>::gather_points(Point3dWPtrVecCC<T> &points)
{
    for (const auto &root_val : roots)
    {
        if (root_val)
            root_val->gather_points(points);
    }
}

template class Octree<double>;
template class Octree<float>;
