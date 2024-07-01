#include "map_storage/sections/sub_tree/base.hpp"
#include <algorithm>
#include <stack>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>

template <typename T>
OctreeNode<T>::OctreeNode(size_t _max_points, bool track_stats)
    : max_points(_max_points), is_leaf(true),
      bbox(std::make_shared<BBox<T>>(track_stats)) {}

template <typename T>
OctreeNode<T>::OctreeNode(
    const Eigen::Matrix<T, 3, 1> &min,
    const Eigen::Matrix<T, 3, 1> &max,
    size_t _max_points, bool track_stats)
    : max_points(_max_points), is_leaf(true),
      bbox(std::make_shared<BBox<T>>(min, max, track_stats)) {}

template <typename T>
int OctreeNode<T>::get_octant(const Point3dPtr<T> &point) const
{
    int octant = 0;
    Eigen::Matrix<T, 3, 1> center = bbox->center();

    if (point->x() >= center.x())
        octant |= 1;
    if (point->y() >= center.y())
        octant |= 2;
    if (point->z() >= center.z())
        octant |= 4;

    return octant;
}

template <typename T>
size_t OctreeNode<T>::get_max_octant_count() const
{
    // gets the index of maximum element in the count array
    auto max_it = std::max_element(octant_counts.begin(), octant_counts.end());
    return *max_it;
}

template <typename T>
void OctreeNode<T>::create_octant(const Eigen::Matrix<T, 3, 1> &center, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, int octant)
{
    Eigen::Matrix<T, 3, 1> new_min = min;
    Eigen::Matrix<T, 3, 1> new_max = max;

    // calculating new bounding box range
    (octant & 1) ? new_min[0] = center[0] : new_max[0] = center[0];
    (octant & 2) ? new_min[1] = center[1] : new_max[1] = center[1];
    (octant & 4) ? new_min[2] = center[2] : new_max[2] = center[2];

    // new child allocated with bounds set - don't track stats of children
    children[octant] = std::make_shared<OctreeNode<T>>(new_min, new_max, max_points, false);
    octant_counts[octant] = 0;
}

template <typename T>
void OctreeNode<T>::subdivide()
{
    Eigen::Matrix<T, 3, 1> center = bbox->center();
    Eigen::Matrix<T, 3, 1> new_min = bbox->get_min();
    Eigen::Matrix<T, 3, 1> new_max = bbox->get_max();

    // create new octants
    size_t max_rep = get_max_octant_count();
    for (size_t idx = 0; idx < octant_counts.size(); ++idx)
    {
        if (octant_counts[idx] == max_rep)
            create_octant(center, new_min, new_max, idx);
    }

    Point3dPtrVect<T> reminants;
    for (const auto &point : points)
    {
        int point_octant = get_octant(point);
        if (!children[point_octant])
            reminants.push_back(std::move(point));
        else
            children[point_octant]->include_point(point, point_octant);
    }

    // replace points with reminants
    points = std::move(reminants);
}

template <typename T>
void OctreeNode<T>::include_point(const Point3dPtr<T> &point, int octant)
{
    // add to new_points
    bbox->unsafe_min_max_update(point->point);
    ++octant_counts[octant];
    points.push_back(point);
}

template <typename T>
void OctreeNode<T>::insert_point(const Point3dPtr<T> &point)
{
    bbox->update(point->point);
    int octant = get_octant(point);
    {
        boost::shared_lock<boost::shared_mutex> lock_s(mutex);
        if (children[octant])
        {
            children[octant]->insert_point(point);
            return;
        }
    }

    // at some point all octants would have been created and this wouldn't be needed
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    if (children[octant]) // check again
    {
        children[octant]->insert_point(point);
        return;
    }

    // process of creating new octants
    ++octant_counts[octant];
    points.push_back(point);
    if (points.size() > max_points)
    {
        subdivide();
        is_leaf = false;
    }
}

template <typename T>
void OctreeNode<T>::unsafe_insert_point(const Point3dPtr<T> &point)
{
    int octant = get_octant(point);
    bbox->unsafe_min_max_update(point->point);
    if (children[octant])
    {
        children[octant]->unsafe_insert_point(point);
        return;
    }

    // at some point all octants would have been created and this wouldn't be needed
    ++octant_counts[octant];
    points.push_back(point);
    if (points.size() > max_points)
    {
        subdivide();
        is_leaf = false;
    }
}

template <typename T>
void OctreeNode<T>::insert_points(const Point3dPtrVectCC<T> &g_points)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    for (const auto &point : g_points)
        unsafe_insert_point(point);
}

template <typename T>
void OctreeNode<T>::insert_points(const Point3dPtrVect<T> &g_points)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    for (const auto &point : g_points)
        unsafe_insert_point(point);
}

template <typename T>
bool OctreeNode<T>::check_leaf()
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return is_leaf;
}

template <typename T>
bool OctreeNode<T>::empty_children()
{
    return std::none_of(
        children.begin(), children.end(),
        [](const auto &child)
        { return static_cast<bool>(child); });
}

template <typename T>
DeleteManager<T> OctreeNode<T>::outside_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type)
{
    DeleteManager<T> to_del;
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    range_delete(to_del, center, range, DeleteCondition::Outside, del_type);

    return to_del;
}

template <typename T>
DeleteManager<T> OctreeNode<T>::within_range_delete(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type)
{
    DeleteManager<T> to_del;
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    range_delete(to_del, center, range, DeleteCondition::Inside, del_type);
    return to_del;
}

template <typename T>
void OctreeNode<T>::handle_delete(DeleteManager<T> &to_del)
{
    std::stack<OctreeNode<T>::Ptr> node_stack;
    for (auto &child : children)
    {
        if (child)
            node_stack.push(std::move(child));
    }

    while (!node_stack.empty())
    {
        auto current_node = node_stack.top();
        node_stack.pop();

        if (!current_node->points.empty())
        {
            for (auto &point : current_node->points)
                to_del.ptd.push_back(point->point);
        }

        if (current_node->is_leaf)
            continue;

        for (auto &child : current_node->children)
        {
            if (child)
                node_stack.push(std::move(child));
        }

        // delete everything current node
        current_node->points.clear();
        current_node.reset();
    }
}

template <typename T>
void OctreeNode<T>::range_delete(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type)
{
    // status check
    auto del_status = DeleteManager<T>::skip_criteria(cond, del_type, bbox, center, range);
    if (del_status == DeleteStatus::Skip)
    {
        to_del.update(bbox->get_min(), bbox->get_max());
        return;
    }

    AVector3TVec<T> ptd; // point collector
    if (del_status == DeleteStatus::Collapse)
    {
        is_leaf = true;
        for (const auto &point : points) // copy all current points
            to_del.ptd.push_back(point->point);

        // take points from children
        handle_delete(to_del);
        to_del.use_min_max = false;
        bbox->reset(false);
        points.clear();

        return;
    }

    // traverse points within blocks
    T range_sq = range * range;
    Point3dPtrVect<T> saved_points;
    bbox->reset(true);

    Eigen::Matrix<T, 3, 1> min = Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::max();
    Eigen::Matrix<T, 3, 1> max = Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::lowest();
    // handling borderline scenario
    for (auto &point : points)
    {
        if (DeleteManager<T>::point_check(point->point, center, cond, range_sq))
        {
            ptd.push_back(point->point);
            point.reset(); // reset shared pointer
        }
        else
        {
            // do this here to reduce bbox lock
            min = min.cwiseMin(point->point);
            max = max.cwiseMax(point->point);
            saved_points.push_back(std::move(point));
        }
    }

    if (!saved_points.empty())
    {
        // Update delete manager and bounding box info.
        bbox->min_max_update(min, max);
        to_del.update(bbox->get_min(), bbox->get_max());
    }
    else
        to_del.use_min_max = false;

    // move points to delete
    bbox->decrement(ptd);
    points = std::move(saved_points);
    std::move(ptd.begin(), ptd.end(), std::back_inserter(to_del.ptd));
    delete_aggregation(to_del, center, range, cond, del_type);
}

template <typename T>
void OctreeNode<T>::delete_aggregation(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type)
{
    // Stack information from each valid child
    std::vector<DeleteManager<T>> locals;
    for (size_t idx = 0; idx < children.size(); ++idx)
    {
        if (children[idx])
        {
            locals.emplace_back();
            children[idx]->range_delete(locals.back(), center, range, cond, del_type);
            if (children[idx]->points.empty() && children[idx]->empty_children())
            {
                children[idx].reset();
                children[idx] = nullptr;
            }
        }
    }

    if (empty_children())
    {
        is_leaf = true;
        if (points.empty())
            bbox->reset(false);
    }

    // update all required information - temp aggregate to reduce lock frequency
    Eigen::Matrix<T, 3, 1> agg_min = Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::max();
    Eigen::Matrix<T, 3, 1> agg_max = Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::lowest();
    AVector3TVec<T> agg_ptd;
    bool update_info = false;

    for (auto &to_del_local : locals)
    {
        if (to_del_local.use_min_max && !is_leaf)
        {
            update_info = true;
            agg_min = agg_min.cwiseMin(to_del_local.min);
            agg_max = agg_max.cwiseMax(to_del_local.max);

            to_del.update(to_del_local.min, to_del_local.max);
        }

        std::move(to_del_local.ptd.begin(), to_del_local.ptd.end(), std::back_inserter(agg_ptd));
    }

    if (update_info)
    {
        bbox->min_max_update(agg_min, agg_max);
        bbox->decrement(agg_ptd);
    }
    std::move(agg_ptd.begin(), agg_ptd.end(), std::back_inserter(to_del.ptd));
}

template <typename T>
void OctreeNode<T>::radius_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k)
{
    // using the helper function to do both
    search_algo(result, qp, range, k);
}

template <typename T>
void OctreeNode<T>::range_search(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range)
{
    // using the helper function to do both
    size_t _k = std::numeric_limits<size_t>::max();
    search_algo(result, qp, range, _k);
}

template <typename T>
void OctreeNode<T>::search_algo(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k)
{
    std::priority_queue<SearchPair, std::vector<SearchPair>, std::greater<>> explore_heap;
    explore_heap.emplace(0.0, this->shared_from_this());
    boost::shared_lock<boost::shared_mutex> lock(mutex);

    T range_sq = range * range;
    while (!explore_heap.empty())
    {
        auto top = explore_heap.top();
        T dist = top.first;
        auto node = top.second;
        explore_heap.pop();

        // skip if point is not within bounding box
        if (BBox<T>::Status::Outside == node->bbox->point_within_bbox(qp, range))
            continue;

        for (const auto &point : node->points)
        {
            T p_dist = (qp - point->point).squaredNorm();
            if (p_dist <= range_sq)
            {
                if (result.size() < k || p_dist < result.top().first)
                {
                    result.emplace(p_dist, point->point);
                    if (result.size() > k)
                        result.pop();
                }
            }
        }

        if (result.size() == k && result.top().first < range_sq)
        {
            range = std::sqrt(result.top().first);
            range_sq = range * range;
        }

        for (const auto &child : node->children)
        {
            if (child)
            {
                T child_dist = child->bbox->closest_distance(qp);
                if (result.size() < k || child_dist <= range_sq)
                    explore_heap.emplace(child_dist, child);
            }
        }
    }
}

template <typename T>
AVector3TVec<T> OctreeNode<T>::get_matched(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, bool verbose)
{
    AVector3TVec<T> matches;
    matches.reserve(result.size());
    std::ostringstream oss;

    if (verbose)
    {
        oss << std::fixed << std::setprecision(4); // Set fixed-point notation and precision
        oss << "\nQuery Point: " << Point3d<T>::eig_to_string(qp) << "\n";
        oss << "Matched Points:\n";
        oss << std::setw(10) << std::left << "Index" << std::setw(40) << "Point" << std::setw(15) << "Distance" << "\n";
        oss << "--------------------------------------------------------------\n";
    }

    int index = 1;
    while (!result.empty())
    {
        auto match = result.top();
        result.pop();

        matches.push_back(match.second);

        if (verbose)
        {
            const Eigen::Matrix<T, 3, 1> &point = match.second;
            T distance = std::sqrt(match.first); // Assuming the stored distance is squared
            oss << std::setw(10) << std::left << index
                << std::setw(40) << Point3d<T>::eig_to_string(point)
                << std::setw(15) << distance << "\n";
        }
        ++index;
    }

    if (verbose)
        std::cout << oss.str();

    return matches;
}

template <typename T>
void OctreeNode<T>::gather_points(Point3dWPtrVecCC<T> &g_points)
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    std::stack<OctreeNode<T>::Ptr> node_stack;
    node_stack.push(this->shared_from_this());

    while (!node_stack.empty())
    {
        auto current_node = node_stack.top();
        node_stack.pop();

        // copy into points folder
        std::copy(
            current_node->points.begin(),
            current_node->points.end(),
            std::back_inserter(g_points));

        if (current_node->is_leaf)
            continue;

        for (auto &child : current_node->children)
        {
            if (child)
                node_stack.push(child);
        }
    }
}

template class OctreeNode<double>;
template class OctreeNode<float>;