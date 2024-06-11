#ifndef OPERATION_LOGGER_HPP
#define OPERATION_LOGGER_HPP

#include "map_storage/sections/sub_tree/octtree.hpp"
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/bbox.hpp"
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <tbb/concurrent_vector.h>

template <typename T>
struct RunningStats
{
    /*** For calculating axis information when needed ****/
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<T, 3, 1> sum = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 1> sum_sq = Eigen::Matrix<T, 3, 1>::Zero();
    T count = 0.0;
    T max_var_coeff = -1.0; // negative means invalid

    RunningStats() = default;

    RunningStats(const Eigen::Matrix<T, 3, 1> &sum, const Eigen::Matrix<T, 3, 1> &sum_sq);

    RunningStats(const Eigen::Matrix<T, 3, 1> &sum, const Eigen::Matrix<T, 3, 1> &sum_sq, const T &count);

    void add_info(const Eigen::Matrix<T, 3, 1> &val);

    T max_variance_val();

    int get_axis() const;
};

// ........... OPERATIONS LOGGER STUFF ...........

enum class OperationType
{
    Insert,
    Delete
};

template <typename T>
struct OperationBase
{
    using Ptr = std::shared_ptr<OperationBase<T>>;
    OperationType type;
    explicit OperationBase(OperationType op_type) : type(op_type) {}
    virtual ~OperationBase() = default;
};
template <typename T>
using OpBasePtr = typename OperationBase<T>::Ptr;

// ........... INSERT OPERATION STUFF ...........
template <typename T>
struct InsertOp : public OperationBase<T>
{
    using Ptr = std::shared_ptr<InsertOp<T>>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3i vox_to_insert;
    RunningStats<T> stats;

    InsertOp(const RunningStats<T> &pth, const Eigen::Vector3i &n_block);

    static Ptr cast(OpBasePtr<T> &op);
};
template <typename T>
using InsertOpPtr = typename InsertOp<T>::Ptr;

// ........... DELETE OPERATION STUFF ...........
template <typename T>
struct DeleteOp : public OperationBase<T>
{
    using Ptr = std::shared_ptr<DeleteOp<T>>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<T, 3, 1> point;
    DeleteCondition cond;
    DeleteType del_type;
    T range;

    DeleteOp(const Eigen::Matrix<T, 3, 1> &point, T range, DeleteCondition cond, DeleteType del_type);

    static Ptr cast(OpBasePtr<T> &op);
};

template <typename T>
using DeleteOpPtr = typename DeleteOp<T>::Ptr;

// ........... OPERATION STACKER STUFF ...........
template <typename T>
struct OperationLogger
{
    /*---- Tracks blocks operations to complete during the rebuilding process ---- */
    using Ptr = std::shared_ptr<OperationLogger<T>>;
    using OpPtr = std::pair<OperationType, OpBasePtr<T>>;
    using S_Vec = tbb::concurrent_vector<OpPtr>;

    void new_slot();

    void log_insert(const RunningStats<T> &pth, const Eigen::Vector3i &n_block);

    void log_delete(const Eigen::Matrix<T, 3, 1> &point, T range, DeleteCondition cond, DeleteType del_type);

    S_Vec get_operations();

    size_t num_operations_left();

private:
    std::vector<S_Vec> ops;
    boost::shared_mutex data;
};

template <typename T>
using OperationLoggerPtr = typename OperationLogger<T>::Ptr;

template <typename T>
using OperationLog = typename OperationLogger<T>::S_Vec;
#endif