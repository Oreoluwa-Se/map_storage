#ifndef KD_TREE_NODE_HPP
#define KD_TREE_NODE_HPP

#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/vox_hash.hpp"
#include <array>
#include <boost/thread/shared_mutex.hpp>
#include <stdexcept>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include "logger.hpp"

enum class Connection
{
    Left,
    Right,
    Parent,
    Remap,
    Previous
};

enum class NodeStatus
{
    None,
    Connected,
    Deleted,
    Rebalancing
};

enum class ClonedStatus
{
    Yes,
    No
};

enum class ScapegoatStatus
{
    Yes,
    No
};

enum class CheckAttribute
{
    TreeSize,
    BoxSize,
    Deleted,
    Axis
};

enum class BoolChecks
{
    Deleted,
    Scapegoat
};

template <typename T>
struct SubtreeAgg
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t num_deleted = 0;
    size_t sub_size = 0;
    Eigen::Matrix<T, 3, 1> v_min, v_max;
};

template <typename T>
class Block : public std::enable_shared_from_this<Block<T>>
{
    static std::atomic<size_t> block_counter; // block id number

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Block<T>>;
    using WPtr = std::weak_ptr<Block<T>>;
    using SearchPair = std::pair<T, Ptr>;

    // ................. SEARCH STRUCT .................
    struct ComparePairs
    {
        bool operator()(const std::pair<T, Ptr> &p1,
                        const std::pair<T, Ptr> &p2)
        {
            return p1.first < p2.first;
        }
    };

    using SearchHeap = std::priority_queue<SearchPair, std::vector<SearchPair>, ComparePairs>;

    Block(const Eigen::Vector3i &value = Eigen::Vector3i::Zero(), int max_vox_size = -1, size_t max_points_oct_layer = 30, T voxel_size = 1.0, bool track_stats = false);

    void set_child(const Ptr &vox, Connection child, int axis = -1);

    Ptr get_child(Connection child);

    void set_aux_connection(const Ptr &vox, Connection child);

    Ptr get_aux_connection(Connection child);

    void set_axis(int axis);

    int get_axis();

    void set_status(NodeStatus status);

    NodeStatus get_status();

    void update_subtree_info();

    bool is_leaf(Ptr &left_child, Ptr &right_child);

    bool is_leaf();

    bool point_insert_clause();

    std::string format_block_info(bool is_left, size_t depth);

    std::string print_subtree();

    size_t get_tree_size();

    bool go_left(const Ptr &comp_block, int axis = -1);

    bool go_left(const Ptr &comp_block, const RunningStats<T> &c_stats);

    bool go_left(const Eigen::Vector3i &comp_block, int axis = -1);

    Ptr insert_or_move(Ptr &new_block, const RunningStats<T> &c_stats);

    bool scapegoat_handler(const T &imbalance_factor, const T &deleted_nodes_imbalance);

    bool check_attributes(BoolChecks check);

    T closest_distance(const Eigen::Matrix<T, 3, 1> &point);

    BBoxStatus<T> point_within_bbox(const Eigen::Matrix<T, 3, 1> &point, T range);

    MinMaxHolder<T> get_min_max();

    // ....................... Logger Operations .......................
    OperationLoggerPtr<T> get_logger();

    Ptr log_insert(const RunningStats<T> &pth, const Eigen::Vector3i &n_block);

    void log_delete(const Eigen::Matrix<T, 3, 1> &point, T range, DeleteCondition cond, DeleteType del_type);

    void detach();

    Ptr clone_block();

    bool cloned_status();

    bool rebal_status();

    std::string bbox_info_to_string();

public:
    OctreePtr<T> oct = nullptr;        // stores the points in current voxel
    Eigen::Vector3i node_rep;          // node voxel number [integer]
    Eigen::Matrix<T, 3, 1> node_rep_d; // node voxel number [decimal]
    size_t block_id;
    mutable boost::shared_mutex mutex;

private:
    void update_subtree_info_helper(Ptr &blk);

    SubtreeAgg<T> subtree_agg(Ptr &l_blk, Ptr &r_blk);

    bool scapegoat_check(const T &imbalance_factor, const T &deleted_nodes_imbalance);

private:
    Ptr left = nullptr;  // left child
    Ptr right = nullptr; // right child
    WPtr parent;         // parent
    WPtr re_map;         // in the case we rebalance the node
    WPtr previous;       // in the case we rebalance the node
    OperationLoggerPtr<T> logger = nullptr;

    Eigen::Matrix<T, 3, 1> v_min, v_max; // voxel boundaries
    T voxel_size;
    int max_vox_size;

    size_t tree_size = 1, num_deleted = 0; // number of voxels within subtree from node
    int axis = -1;
    bool require_update = true;
    NodeStatus status = NodeStatus::None;
    ClonedStatus c_status = ClonedStatus::No;
    ScapegoatStatus s_status = ScapegoatStatus::No;

    // lock
    mutable boost::shared_mutex axis_mutex, remap_mutex;
    mutable boost::shared_mutex status_mutex, logger_mutex;
    mutable boost::shared_mutex left_mutex, right_mutex, wconnect_mutex;
};

template <typename T>
using BlockPtr = typename Block<T>::Ptr;

template <typename T>
using BlockWPtr = typename Block<T>::WPtr;

template <typename T>
using BlockWPtrVec = std::vector<BlockWPtr<T>>;

template <typename T>
using BlockPtrVec = std::vector<BlockPtr<T>>;

template <typename T>
using BlockWPtrVecCC = tbb::concurrent_vector<BlockWPtr<T>>;

template <typename T>
using BlockPtrVecCC = tbb::concurrent_vector<BlockPtr<T>>;

template <typename T>
using BlockPtrMap = tbb::concurrent_hash_map<Eigen::Vector3i, BlockPtr<T>, VoxelHash>;

template <typename T>
using BlockWPtrMap = tbb::concurrent_hash_map<Eigen::Vector3i, BlockWPtr<T>, VoxelHash>;

template <typename T>
using BlockPointPair = std::pair<BlockPtr<T>, Eigen::Matrix<T, 3, 1>>;

#endif