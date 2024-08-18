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
void OctreeNode<T>::create_child(const Eigen::Matrix<T, 3, 1> &center, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, int loc)
{
    Eigen::Matrix<T, 3, 1> new_min = min;
    Eigen::Matrix<T, 3, 1> new_max = max;

    // calculating new bounding box range
    (loc & 1) ? new_min[0] = center[0] : new_max[0] = center[0];
    (loc & 2) ? new_min[1] = center[1] : new_max[1] = center[1];
    (loc & 4) ? new_min[2] = center[2] : new_max[2] = center[2];

    // new child allocated with bounds set - don't track stats of children
    children[loc] = std::make_shared<OctreeNode<T>>(new_min, new_max, max_points, false);
    children[loc]->depth = (depth + 1) % 3;
}

template <typename T>
void OctreeNode<T>::subdivide()
{
    // establish midpoint, split and clear points from current
    split_center = bbox->center();
    Eigen::Matrix<T, 3, 1> new_min = bbox->get_min();
    Eigen::Matrix<T, 3, 1> new_max = bbox->get_max();

    create_child(split_center, new_min, new_max, 0);
    create_child(split_center, new_min, new_max, 1);

    for (auto &point : points)
    {
        if (move_left(point->point))
            children[0]->include_point(std::move(point));
        else
            children[1]->include_point(std::move(point));
    }

    // replace points with reminants
    points.clear();
}

template <typename T>
void OctreeNode<T>::include_point(Point3dPtr<T> &&point)
{
    // add to new_points
    bbox->unsafe_min_max_update(point->point);
    ++curr_size;
    points.push_back(std::move(point));
}

template <typename T>
bool OctreeNode<T>::move_left(const Eigen::Matrix<T, 3, 1> &pt)
{
    // split_center is defined when we subdivide the vector
    int axis = depth % 3;
    return pt[axis] < split_center[axis];
}

template <typename T>
void OctreeNode<T>::insert_point(const Point3dPtr<T> &point)
{
    std::stack<typename OctreeNode<T>::Ptr> stack;
    stack.push(this->shared_from_this());

    while (!stack.empty())
    {
        auto curr_node = stack.top();
        stack.pop();

        curr_node->bbox->update(point->point);
        {
            boost::shared_lock<boost::shared_mutex> lock(curr_node->mutex);
            if (!curr_node->is_leaf)
            {
                if (curr_node->move_left(point->point))
                    stack.push(curr_node->children[0]);
                else
                    stack.push(curr_node->children[1]);

                continue;
            }

            ++curr_node->curr_size;
            curr_node->points.push_back(point);
            if (curr_node->curr_size.load(std::memory_order_relaxed) < curr_node->max_points)
                break; // inserted and we are done
        }

        boost::unique_lock<boost::shared_mutex> lock(curr_node->mutex);
        if (!curr_node->is_leaf)
        {
            if (curr_node->move_left(point->point))
                stack.push(curr_node->children[0]);
            else
                stack.push(curr_node->children[1]);
        }

        curr_node->is_leaf = false;
        curr_node->subdivide();
        if (curr_node->move_left(point->point))
            stack.push(curr_node->children[0]);
        else
            stack.push(curr_node->children[1]);
    }
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
                to_del.ptd.push_back(std::move(point->point));
        }

        if (!current_node->is_leaf)
        {
            for (auto &child : current_node->children)
            {
                if (child)
                    node_stack.push(std::move(child));
            }
        }

        // delete everything current node
        current_node->points.clear();
        current_node.reset();
    }
}

template <typename T>
bool OctreeNode<T>::skippable_del_node_ops(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type)
{
    auto del_status = DeleteManager<T>::skip_criteria(cond, del_type, bbox, center, range);
    if (del_status == DeleteStatus::Skip)
    {
        to_del.update(bbox->get_min(), bbox->get_max());
        return true;
    }

    // this implies we can essentially remove everything from this section
    if (del_status == DeleteStatus::Collapse)
    {
        is_leaf = true;
        for (const auto &point : points)
            to_del.ptd.push_back(point->point);

        // take points from children
        handle_delete(to_del);
        bbox->reset(false);
        points.clear();

        return true;
    }

    return false;
}

template <typename T>
void OctreeNode<T>::top_down_update_bbox_info()
{
    Eigen::Matrix<T, 3, 1> min = Eigen::Matrix<T, 3, 1>::Ones() * std::numeric_limits<T>::max();
    Eigen::Matrix<T, 3, 1> max = Eigen::Matrix<T, 3, 1>::Ones() * std::numeric_limits<T>::lowest();
    for (auto &child : children)
    {
        min = min.cwiseMin(child->bbox->get_min());
        max = max.cwiseMax(child->bbox->get_max());
    }

    bbox->min_max_update(min, max);
    split_center = 0.5 * (min + max);
}

template <typename T>
void OctreeNode<T>::leaf_vector_delete(AVector3TVec<T> &ptd, const Eigen::Matrix<T, 3, 1> &center, T range_sq, DeleteCondition cond)
{
    Point3dPtrVectCC<T> saved_points;
    bbox->reset(true);

    Eigen::Matrix<T, 3, 1> min = Eigen::Matrix<T, 3, 1>::Ones() * std::numeric_limits<T>::max();
    Eigen::Matrix<T, 3, 1> max = Eigen::Matrix<T, 3, 1>::Ones() * std::numeric_limits<T>::lowest();

    for (auto &point : points)
    {
        if (DeleteManager<T>::point_check(point->point, center, cond, range_sq))
        {
            ptd.push_back(point->point);
            point.reset(); // reset shared pointer
        }
        else
        {
            min = min.cwiseMin(point->point);
            max = max.cwiseMax(point->point);
            saved_points.push_back(std::move(point));
        }
    }

    if (saved_points.empty())
    {
        points.clear();
        return;
    }

    // outgoing
    points = saved_points;
    bbox->min_max_update(min, max);

    // adjust mean cov values
    bbox->decrement(ptd);
}

template <typename T>
void OctreeNode<T>::range_delete(DeleteManager<T> &to_del, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type)
{
    using NodeMarkerPair = std::pair<typename OctreeNode<T>::Ptr, bool>;
    std::stack<NodeMarkerPair> stack;
    stack.push({this->shared_from_this(), false});

    // traverse points within blocks
    T range_sq = range * range;
    Eigen::Matrix<T, 3, 1> min, max;

    while (!stack.empty())
    {
        auto s_info = stack.top();
        auto curr_node = s_info.first;
        bool visited = s_info.second;
        stack.pop();

        // here either we don't delete subtree or we remove the entire subtree
        if (curr_node->skippable_del_node_ops(to_del, center, range, cond, del_type))
            continue;

        if (visited)
        {
            if (curr_node->is_leaf)
            {
                // mostly for intermediate cases and to update bbox information
                AVector3TVec<T> ptd;
                curr_node->leaf_vector_delete(ptd, center, range_sq, cond);
                to_del.update(curr_node->bbox->get_min(), curr_node->bbox->get_max());

                if (curr_node->points.empty())
                    curr_node.reset(); // deallocation

                std::move(ptd.begin(), ptd.end(), std::back_inserter(to_del.ptd));
                continue;
            }
            // case where children are empty and current is empty remove it
            else if (curr_node->empty_children())
            {
                curr_node.reset();
                continue;
            }
            else
            {
                if (curr_node->children[1] && curr_node->children[0])
                {
                    // update current with left and right bboxes;
                    curr_node->top_down_update_bbox_info();
                    continue;
                }

                // Move up the valid child if only one child exists
                auto valid_child = curr_node->children[1] ? curr_node->children[1] : curr_node->children[0];

                curr_node->is_leaf = true;
                curr_node->bbox = valid_child->bbox;
                curr_node->points = valid_child->points;
                curr_node->curr_size.store(curr_node->points.size());

                // reset valid_child
                valid_child.reset();
            }
        }
        else
        {
            // push current and children
            stack.push({curr_node, true});

            if (!curr_node->is_leaf)
            {
                if (curr_node->children[1])
                    stack.push({curr_node->children[1], false});
                if (curr_node->children[0])
                    stack.push({curr_node->children[0], false});
            }
        }
    }
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
void OctreeNode<T>::process_leaf_node(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, T &range_sq, size_t k)
{
    for (const auto &point : points)
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
}

template <typename T>
void OctreeNode<T>::search_algo(SearchHeap<T> &result, const Eigen::Matrix<T, 3, 1> &qp, T &range, size_t k)
{
    std::priority_queue<SearchPair, std::vector<SearchPair>, std::greater<>> explore_heap;
    explore_heap.emplace(0.0, this->shared_from_this());

    T range_sq = range * range;

    boost::shared_lock<boost::shared_mutex> lock(mutex);
    while (!explore_heap.empty())
    {
        auto top = explore_heap.top();
        auto node = top.second;
        explore_heap.pop();

        // skip if point is not within bounding box
        if (BBox<T>::Status::Outside == node->bbox->point_within_bbox(qp, range))
            continue;

        if (node->check_leaf())
        {
            node->process_leaf_node(result, qp, range, range_sq, k);
            continue;
        }

        // finding next search
        bool left_check = move_left(qp);
        auto near = left_check ? node->children[0] : node->children[1];
        if (near)
        {
            auto nd = near->bbox->closest_distance(qp);
            explore_heap.emplace(near->bbox->closest_distance(qp), near);
        }

        auto further = left_check ? node->children[1] : node->children[0];
        if (further)
        {
            if (BBox<T>::Status::Outside != further->bbox->point_within_bbox(qp, range))
                explore_heap.emplace(further->bbox->closest_distance(qp), further);
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
    std::stack<OctreeNode<T>::Ptr> node_stack;
    node_stack.push(this->shared_from_this());

    while (!node_stack.empty())
    {
        auto current_node = node_stack.top();
        node_stack.pop();

        // copy into points folder
        if (current_node->check_leaf())
        {
            boost::shared_lock<boost::shared_mutex> lock(current_node->mutex);
            std::copy(
                current_node->points.begin(),
                current_node->points.end(),
                std::back_inserter(g_points));
        }

        for (auto &child : current_node->children)
        {
            if (child)
                node_stack.push(child);
        }
    }
}

template class OctreeNode<double>;
template class OctreeNode<float>;