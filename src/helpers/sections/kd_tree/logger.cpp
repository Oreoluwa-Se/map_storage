#include "map_storage/sections/kd_tree/logger.hpp"

template <typename T>
RunningStats<T>::RunningStats(const Eigen::Matrix<T, 3, 1> &sum, const Eigen::Matrix<T, 3, 1> &sum_sq)
    : sum(sum), sum_sq(sum_sq) {}

template <typename T>
RunningStats<T>::RunningStats(const Eigen::Matrix<T, 3, 1> &sum, const Eigen::Matrix<T, 3, 1> &sum_sq, const T &count)
    : sum(sum), sum_sq(sum_sq), count(count) {}

template <typename T>
void RunningStats<T>::add_info(const Eigen::Matrix<T, 3, 1> &val)
{
    sum.noalias() += val;
    sum_sq.noalias() += val.cwiseProduct(val);
    count += 1.0;
}

template <typename T>
T RunningStats<T>::max_variance_val()
{
    if (max_var_coeff >= T(0.0))
        return max_var_coeff;

    T max_val = T(0);

    if (count >= T(3.0))
    {
        Eigen::Matrix<T, 3, 1> mean = sum / count;
        Eigen::Matrix<T, 3, 1> var = (sum_sq / count) - mean.cwiseProduct(mean);
        max_val = std::abs(var.maxCoeff());
    }

    max_var_coeff = max_val;

    return max_val;
}

template <typename T>
int RunningStats<T>::get_axis() const
{
    Eigen::Matrix<T, 3, 1> mean = sum / count;
    Eigen::Matrix<T, 3, 1> var = (sum_sq / count) - mean.cwiseProduct(mean);

    // axis decider
    int calc_axis = 0;
    if (var(1) > var(calc_axis))
        calc_axis = 1;
    if (var(2) > var(calc_axis))
        calc_axis = 2;

    return calc_axis;
}

// Allowed types
template struct RunningStats<double>;
template struct RunningStats<float>;

// ........... OPERATIONS LOGGER STUFF ...........
template <typename T>
InsertOp<T>::InsertOp(const RunningStats<T> &pth, const Eigen::Vector3i &n_block)
    : OperationBase<T>(OperationType::Insert, true),
      stats(pth.sum, pth.sum_sq, pth.count),
      vox_to_insert(n_block) {}

template <typename T>
typename InsertOp<T>::Ptr InsertOp<T>::cast(OpBasePtr<T> &op)
{
    if (!op)
        return nullptr;

    if (auto casted = std::dynamic_pointer_cast<InsertOp<T>>(op))
        return casted;

    return nullptr;
}

template struct InsertOp<double>;
template struct InsertOp<float>;

// ........... DELETE OPERATION STUFF ...........
template <typename T>
DeleteOp<T>::DeleteOp(const Eigen::Matrix<T, 3, 1> &point, T range, DeleteCondition cond, DeleteType del_type)
    : OperationBase<T>(OperationType::Delete, true), point(point), cond(cond), del_type(del_type), range(range) {}

template <typename T>
typename DeleteOp<T>::Ptr DeleteOp<T>::cast(OpBasePtr<T> &op)
{
    if (!op)
        return nullptr;

    if (auto casted = std::dynamic_pointer_cast<DeleteOp<T>>(op))
        return casted;

    return nullptr;
}

template struct DeleteOp<double>;
template struct DeleteOp<float>;

// ........... OPERATION STACKER STUFF ...........

template <typename T>
void OperationLogger<T>::log_insert(const RunningStats<T> &pth, const Eigen::Vector3i &n_block)
{
    ops.emplace(OperationType::Insert, std::make_shared<InsertOp<T>>(pth, n_block));
    ++ops_left;
}

template <typename T>
void OperationLogger<T>::log_delete(const Eigen::Matrix<T, 3, 1> &point, T range, DeleteCondition cond, DeleteType del_type)
{

    ops.emplace(OperationType::Delete, std::make_shared<DeleteOp<T>>(point, range, cond, del_type));
    ++ops_left;
}

template <typename T>
typename OperationLogger<T>::OpPtr OperationLogger<T>::get_operations()
{
    OpPtr op;
    if (ops.try_pop(op))
    {
        --ops_left;
        return op;
    }

    return op;
}

template <typename T>
size_t OperationLogger<T>::num_operations_left()
{
    return ops_left.load(std::memory_order_relaxed);
}

template struct OperationLogger<double>;
template struct OperationLogger<float>;
