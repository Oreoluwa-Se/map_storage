#include "map_storage/sections/kd_tree/node.hpp"
#include <iomanip>
#include <sstream>
#include <stack>
#include <tbb/parallel_invoke.h>
#include <tuple>

template <typename T>
std::atomic<size_t> Block<T>::block_counter{0};

// Individual blocks
template <typename T>
Block<T>::Block(const Eigen::Vector3i &value, int max_vox_size, size_t max_points_oct_layer, T voxel_size, bool track_stats)
    : oct(std::make_shared<Octree<T>>(max_points_oct_layer, track_stats, max_vox_size)),
      node_rep(value), node_rep_d(value.cast<T>()), block_id(++block_counter),
      v_min(node_rep_d - Eigen::Matrix<T, 3, 1>::Constant(voxel_size * 0.5)),
      v_max(node_rep_d + Eigen::Matrix<T, 3, 1>::Constant(voxel_size * 0.5)),
      voxel_size(voxel_size), max_vox_size(max_vox_size) {}

template <typename T>
void Block<T>::set_axis(int _axis)
{
    boost::unique_lock<boost::shared_mutex> lock(axis_mutex);
    axis = _axis;
}

template <typename T>
void Block<T>::set_status(NodeStatus _status)
{
    boost::unique_lock<boost::shared_mutex> lock(status_mutex);
    status = _status;
}

template <typename T>
void Block<T>::set_child(const typename Block<T>::Ptr &vox, Connection child, int _axis)
{
    if (child == Connection::Left)
    {
        boost::unique_lock<boost::shared_mutex> lock(left_mutex);
        left = vox;
    }
    else if (child == Connection::Right)
    {
        boost::unique_lock<boost::shared_mutex> lock(right_mutex);
        right = vox;
    }

    if (_axis != -1)
    {
        boost::unique_lock<boost::shared_mutex> lock(axis_mutex);
        axis = _axis;
    }
}

template <typename T>
typename Block<T>::Ptr Block<T>::get_child(Connection child)
{
    if (child == Connection::Left)
    {
        boost::shared_lock<boost::shared_mutex> lock(left_mutex);
        return left;
    }
    else if (child == Connection::Right)
    {
        boost::shared_lock<boost::shared_mutex> lock(right_mutex);
        return right;
    }

    return nullptr;
}

template <typename T>
void Block<T>::set_aux_connection(const typename Block<T>::Ptr &vox, Connection child)
{
    if (child == Connection::Parent)
    {
        boost::unique_lock<boost::shared_mutex> lock(wconnect_mutex);
        parent = vox;
    }
    else if (child == Connection::Remap)
    {
        boost::unique_lock<boost::shared_mutex> lock(remap_mutex);
        re_map = vox;
    }
    else if (child == Connection::Previous)
    {
        boost::unique_lock<boost::shared_mutex> lock(wconnect_mutex);
        previous = vox;
    }
}

template <typename T>
typename Block<T>::Ptr Block<T>::get_aux_connection(Connection child)
{
    typename Block<T>::Ptr sp;

    if (child == Connection::Parent)
    {
        boost::shared_lock<boost::shared_mutex> lock(wconnect_mutex);
        sp = parent.lock();
    }
    else if (child == Connection::Remap)
    {
        boost::shared_lock<boost::shared_mutex> lock(remap_mutex);
        sp = re_map.lock();
    }
    else if (child == Connection::Previous)
    {
        boost::shared_lock<boost::shared_mutex> lock(wconnect_mutex);
        sp = previous.lock();
    }

    return sp;
}

template <typename T>
SubtreeAgg<T> Block<T>::subtree_agg(Ptr &l_blk, Ptr &r_blk)
{
    SubtreeAgg<T> sub_stats;
    bool min_max_checked = false;

    for (const auto &blk : {l_blk, r_blk})
    {
        if (!blk)
            continue;

        boost::shared_lock<boost::shared_mutex> lock(blk->mutex);
        sub_stats.sub_size += blk->tree_size;
        sub_stats.num_deleted += blk->num_deleted;

        if (!min_max_checked)
        {
            min_max_checked = true;
            sub_stats.v_min = blk->v_min;
            sub_stats.v_max = blk->v_max;
        }
        else
        {
            sub_stats.v_min = sub_stats.v_min.cwiseMin(blk->v_min);
            sub_stats.v_max = sub_stats.v_max.cwiseMax(blk->v_max);
        }
    }

    return sub_stats;
}

template <typename T>
BBoxStatus<T> Block<T>::point_within_bbox(const Eigen::Matrix<T, 3, 1> &point, T range)
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return BBox<T>::point_within_bbox(point, v_min, v_max, range);
}

template <typename T>
bool Block<T>::is_leaf(typename Block<T>::Ptr &left_child, typename Block<T>::Ptr &right_child)
{
    // update with local on the down stream
    left_child = get_child(Connection::Left);
    right_child = get_child(Connection::Right);

    return left_child == nullptr && right_child == nullptr;
}

template <typename T>
bool Block<T>::is_leaf()
{
    // update with local on the down stream
    auto left_child = get_child(Connection::Left);
    auto right_child = get_child(Connection::Right);

    return left_child == nullptr && right_child == nullptr;
}

template <typename T>
void Block<T>::update_subtree_info()
{
    { // trigger that we are updating current
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        require_update = true;
    }

    auto curr_blk = this->shared_from_this();
    update_subtree_info_helper(curr_blk);
}

template <typename T>
void Block<T>::update_subtree_info_helper(typename Block<T>::Ptr &blk)
{
    if (!blk)
        return;

    {
        boost::shared_lock<boost::shared_mutex> lock(blk->mutex);
        if (!blk->require_update)
            return;
    }

    // update with local on the down stream
    Ptr left_blk, right_blk;
    NodeStatus l_status;

    if (blk->is_leaf(left_blk, right_blk))
    {
        // get current node status
        {
            boost::shared_lock<boost::shared_mutex> lock_s(blk->status_mutex);
            l_status = blk->status;
        }

        boost::unique_lock<boost::shared_mutex> lock(blk->mutex);
        blk->num_deleted = size_t(l_status == NodeStatus::Deleted);
        blk->tree_size = size_t(l_status != NodeStatus::Deleted);
        blk->require_update = false;
        return;
    }

    if (left_blk && right_blk)
    {
        tbb::parallel_invoke(
            [&]
            { update_subtree_info_helper(left_blk); },
            [&]
            { update_subtree_info_helper(right_blk); });
    }
    else
    {
        if (left_blk)
            update_subtree_info_helper(left_blk);

        else if (right_blk)
            update_subtree_info_helper(right_blk);
    }

    SubtreeAgg<T> agg = subtree_agg(left_blk, right_blk);
    // get current node status
    {
        boost::shared_lock<boost::shared_mutex> lock_s(blk->status_mutex);
        l_status = status;
    }

    boost::unique_lock<boost::shared_mutex> lock(blk->mutex);
    {
        // update info locally
        blk->tree_size = size_t(l_status != NodeStatus::Deleted) + agg.sub_size;
        blk->num_deleted = size_t(l_status == NodeStatus::Deleted) + agg.num_deleted;
        blk->v_min = v_min.cwiseMin(agg.v_min);
        blk->v_max = v_max.cwiseMax(agg.v_max);
        blk->require_update = false;
    }
}

template <typename T>
MinMaxHolder<T> Block<T>::get_min_max()
{
    MinMaxHolder<T> mm;
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    mm.min = v_min;
    mm.max = v_max;

    return mm;
}

template <typename T>
std::string Block<T>::format_block_info(bool is_left, size_t depth)
{
    std::ostringstream n_string;
    if (depth > 0)
        n_string << (is_left ? "├───L " : "└───R ");

    n_string << std::fixed << std::setprecision(1);

    // Block information
    n_string << "[" << node_rep_d[0] << ", " << node_rep_d[1] << ", " << node_rep_d[2] << "] ";
    n_string << "Size: " << get_tree_size() << " | "
             << "Axis: " << get_axis() << " | "
             << "Box_size: " << oct->bbox->get_size();

    if (check_attributes(BoolChecks::Deleted))
        n_string << " [Deleted]";

    if (check_attributes(BoolChecks::Scapegoat))
        n_string << " [Scapegoat]";

    return n_string.str();
}

template <typename T>
std::string Block<T>::print_subtree()
{
    std::vector<std::tuple<BlockPtr<T>, int, bool>> stack;
    stack.reserve(10);

    stack.emplace_back(this->shared_from_this(), 0, false);

    std::ostringstream tree_output;
    size_t total_points_size = 0;
    size_t total_nodes = 0;
    while (!stack.empty())
    {
        auto point_depth_pair = stack.back();
        stack.pop_back();

        auto point = std::get<0>(point_depth_pair);
        auto depth = std::get<1>(point_depth_pair);
        auto is_left = std::get<2>(point_depth_pair);

        // Indentation for the tree structure.
        tree_output << std::string(depth * 4, ' ');

        // Add formatted block info.
        tree_output << point->format_block_info(is_left, depth) << "\n";
        if (!point->check_attributes(BoolChecks::Deleted))
        {
            total_points_size += point->oct->bbox->get_size();
            ++total_nodes;
        }

        // Add children to the stack.
        if (auto sp = point->get_child(Connection::Right))
            stack.emplace_back(sp, depth + 1, false);

        if (auto sp = point->get_child(Connection::Left))
            stack.emplace_back(sp, depth + 1, true);
    }

    std::ostringstream out;
    out << "Number of points in map: " << total_points_size << "\n";
    out << "Number of current voxels: " << total_nodes << std::endl;
    out << " =========================== " << std::endl;
    out << tree_output.str();

    return out.str();
}

template <typename T>
NodeStatus Block<T>::get_status()
{
    boost::shared_lock<boost::shared_mutex> lock(status_mutex);
    return status;
}

template <typename T>
size_t Block<T>::get_tree_size()
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return tree_size;
}

template <typename T>
int Block<T>::get_axis()
{
    boost::shared_lock<boost::shared_mutex> lock(axis_mutex);
    return axis;
}

template <typename T>
bool Block<T>::go_left(const typename Block<T>::Ptr &comp_block, int _axis)
{
    auto m_axis = _axis;
    if (m_axis == -1)
        m_axis = get_axis();

    return comp_block->node_rep_d[m_axis] < node_rep_d[m_axis];
}

template <typename T>
bool Block<T>::go_left(const Eigen::Vector3i &comp_block, int _axis)
{
    auto m_axis = _axis;
    if (m_axis == -1)
        m_axis = get_axis();

    return comp_block[m_axis] < node_rep[m_axis];
}

template <typename T>
bool Block<T>::go_left(const Ptr &comp_block, const RunningStats<T> &c_stats)
{
    int c_axis = get_axis();
    if (c_axis != -1)
        return comp_block->node_rep[c_axis] < node_rep[c_axis];

    {
        boost::unique_lock<boost::shared_mutex> lock(axis_mutex);
        if (axis == -1)
        {
            // calculate new axis
            axis = c_stats.get_axis();
            c_axis = axis;
        }
    }

    return comp_block->node_rep_d[c_axis] < node_rep_d[c_axis];
}

template <typename T>
void Block<T>::swap_locations(typename Block<T>::Ptr &comp_block, typename Block<T>::Ptr &to_swap)
{
    boost::unique_lock<boost::shared_mutex> lock_l(left_mutex, boost::defer_lock);
    boost::unique_lock<boost::shared_mutex> lock_r(right_mutex, boost::defer_lock);
    boost::lock(lock_l, lock_r);

    if (left && left->node_rep == comp_block->node_rep)
        left = to_swap;
    else if (right && right->node_rep == comp_block->node_rep)
        right = to_swap;
    else
        throw std::logic_error("No matching block found to swap locations.");
}

template <typename T>
void Block<T>::voxel_increment_info_update(typename Block<T>::Ptr &new_block)
{
    // increment the tree size and update boundary information
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    v_min = v_min.cwiseMin(new_block->node_rep_d);
    v_max = v_max.cwiseMax(new_block->node_rep_d);
    ++tree_size;
}

template <typename T>
typename Block<T>::Ptr Block<T>::insert_or_move(typename Block<T>::Ptr &new_block, const RunningStats<T> &c_stats)
{
    // find direction and assign axis if needed
    voxel_increment_info_update(new_block);

    auto next_dir = go_left(new_block, c_stats) ? Connection::Left : Connection::Right;
    auto &cmtx = (next_dir == Connection::Left) ? left_mutex : right_mutex;

    {
        boost::shared_lock<boost::shared_mutex> lock(cmtx);
        {
            auto &ptr = (next_dir == Connection::Left) ? left : right;
            if (ptr)
            {
                log_insert(c_stats, new_block->node_rep);
                return ptr;
            }
        }
    }

    boost::unique_lock<boost::shared_mutex> u_lock(cmtx);
    auto &ptr = (next_dir == Connection::Left) ? left : right;
    if (!ptr)
    {
        ptr = new_block;
        u_lock.unlock();
        log_insert(c_stats, new_block->node_rep);
        return nullptr;
    }

    u_lock.unlock();
    log_insert(c_stats, new_block->node_rep);
    return ptr;
}

template <typename T>
bool Block<T>::scapegoat_handler(const T &imbalance_factor, const T &deleted_nodes_imbalance)
{
    // if node has been deleted or already marked -> being taken care of already
    if (check_attributes(BoolChecks::Deleted) || check_attributes(BoolChecks::Scapegoat))
        return false;

    // should check the parent aswell - if parent already scapegoat just tag
    if (auto sp = get_aux_connection(Connection::Parent))
    {
        if (sp->check_attributes(BoolChecks::Scapegoat))
        {
            boost::unique_lock<boost::shared_mutex> lock(status_mutex);
            s_status = ScapegoatStatus::Yes;
            return false;
        }
    }

    if (scapegoat_check(imbalance_factor, deleted_nodes_imbalance))
    {
        { // done this way to avoid false positives
            boost::shared_lock<boost::shared_mutex> lock(status_mutex);
            if (s_status == ScapegoatStatus::Yes)
                return false;
        }

        boost::unique_lock<boost::shared_mutex> lock(status_mutex);
        if (s_status == ScapegoatStatus::Yes)
            return false;

        s_status = ScapegoatStatus::Yes;
        return true;
    }

    return false;
}

template <typename T>
bool Block<T>::scapegoat_check(const T &imb_factor, const T &del_imb_factor)
{
    // leaf nodes so we skip the check
    Ptr left_blk, right_blk;
    if (is_leaf(left_blk, right_blk))
        return false;

    T c_tree_size, c_num_invalid;
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        c_tree_size = static_cast<T>(tree_size);
        c_num_invalid = static_cast<T>(num_deleted);
    }

    if (del_imb_factor != 1.0)
    {
        // check number of invalid nodes
        if (c_num_invalid > del_imb_factor * c_tree_size)
            return true;
    }

    if (imb_factor == 1.0)
        return false; // skip imbalance factor

    const T left_size = left_blk ? static_cast<T>(left->get_tree_size()) : 0.0;
    const T right_size = right_blk ? static_cast<T>(right->get_tree_size()) : 0.0;

    // an early exit criteria for trees close to being equal
    if (std::abs(left_size - right_size) <= 1.0)
        return false;

    if (c_tree_size >= 4.0)
    { // Check for imbalance in non-trivial trees.
        const T max_child_size = imb_factor * (c_tree_size - 1.0);
        return left_size > max_child_size || right_size > max_child_size;
    }

    // In smaller trees, one side must be empty.
    return left_size == 0.0 || right_size == 0.0;
}

template <typename T>
bool Block<T>::check_attributes(BoolChecks check)
{
    switch (check)
    {
    case BoolChecks::Deleted:
    {
        boost::shared_lock<boost::shared_mutex> lock(status_mutex);
        return status == NodeStatus::Deleted;
    }
    case BoolChecks::Scapegoat:
    {
        boost::shared_lock<boost::shared_mutex> lock(mutex);
        return s_status == ScapegoatStatus::Yes;
    }
    default:
        return false;
    }
}

template <typename T>
T Block<T>::closest_distance(const Eigen::Matrix<T, 3, 1> &point)
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return BBox<T>::closest_distance(point, v_min, v_max);
}

template <typename T>
std::string Block<T>::bbox_info_to_string()
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3); // Set fixed-point notation and precision
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    oss << "\nVoxel Level BoundingBox Information:" << std::endl;
    oss << "-------------------------" << std::endl;
    oss << "Current: [" << node_rep(0) << ".0, " << node_rep(1) << ".0, " << node_rep(2) << ".0]" << std::endl;
    oss << "Min: [" << v_min(0) << ", " << v_min(1) << ", " << v_min(2) << "]" << std::endl;
    oss << "Max: [" << v_max(0) << ", " << v_max(1) << ", " << v_max(2) << "]" << std::endl;
    oss << "-------------------------" << std::endl;

    return oss.str();
}
// ....................... Logger Operations .......................
template <typename T>
void Block<T>::log_insert(const RunningStats<T> &c_stats, const Eigen::Vector3i &n_block)
{
    if (get_status() != NodeStatus::Rebalancing)
        return;

    // Log the insert operation. At this point, logger is guaranteed to exist.
    auto logger = get_logger(true);
    logger->log_insert(c_stats, n_block);
}

template <typename T>
void Block<T>::log_delete(const Eigen::Matrix<T, 3, 1> &point, T range, DeleteCondition cond, DeleteType del_type)
{
    {
        boost::shared_lock<boost::shared_mutex> lock(logger_mutex);
        if (logger)
        {
            logger->log_delete(point, range, cond, del_type);
            return;
        }
    }

    boost::unique_lock<boost::shared_mutex> lock(logger_mutex);
    if (!logger)
        logger = std::make_shared<OperationLogger<T>>();

    logger->log_delete(point, range, cond, del_type);
}

template <typename T>
OperationLoggerPtr<T> Block<T>::get_logger(bool create)
{
    {
        boost::shared_lock<boost::shared_mutex> lock(logger_mutex);
        if (logger || !create)
            return logger;
    }

    boost::unique_lock<boost::shared_mutex> lock(logger_mutex);
    if (!logger)
        logger = std::make_shared<OperationLogger<T>>();

    return logger;
}

template <typename T>
void Block<T>::detach()
{
    {
        boost::unique_lock<boost::shared_mutex> lock(left_mutex);
        left.reset();
    }

    {
        boost::unique_lock<boost::shared_mutex> lock(right_mutex);
        right.reset();
    }

    {
        boost::unique_lock<boost::shared_mutex> lock(logger_mutex);
        logger = nullptr;
    }
}

template <typename T>
typename Block<T>::Ptr Block<T>::clone_block()
{
    auto blk = std::make_shared<Block<T>>();

    // base information
    blk->node_rep = node_rep;
    blk->node_rep_d = node_rep_d;
    blk->v_min = node_rep_d - Eigen::Matrix<T, 3, 1>::Constant(voxel_size * 0.5);
    blk->v_max = node_rep_d + Eigen::Matrix<T, 3, 1>::Constant(voxel_size * 0.5);
    blk->voxel_size = voxel_size;

    // points under current
    blk->oct = oct;

    // set current status
    {
        boost::unique_lock<boost::shared_mutex> lock(status_mutex);
        c_status = ClonedStatus::Yes;
    }

    return blk;
}

template <typename T>
bool Block<T>::cloned_status()
{
    boost::shared_lock<boost::shared_mutex> lock(status_mutex);
    return c_status == ClonedStatus::Yes;
}

template class Block<double>;
template class Block<float>;