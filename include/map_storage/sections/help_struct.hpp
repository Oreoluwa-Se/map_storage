#ifndef OCTANT_HELP_STRUCTS
#define OCTANT_HELP_STRUCTS
#include "map_storage/utils/alias.hpp"
#include "map_storage/utils/bbox.hpp"
#include <queue>
#include <limits>
#include <algorithm>

enum class DeleteStatus
{
    Skip,
    Collapse,
    None
};

template <typename T>
struct MinMaxHolder
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<T, 3, 1> min, max;
};

template <typename T>
struct DeleteManager
{
    // helps in managing delete process
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix<T, 3, 1> min = Eigen::Matrix<T, 3, 1>::Ones() * std::numeric_limits<T>::max();
    Eigen::Matrix<T, 3, 1> max = Eigen::Matrix<T, 3, 1>::Ones() * std::numeric_limits<T>::lowest();
    AVector3TVec<T> ptd;
    bool use_min_max = true;

    void update(const Eigen::Matrix<T, 3, 1> &point);

    void update(const Eigen::Matrix<T, 3, 1> &point_a, const Eigen::Matrix<T, 3, 1> &point_b);

    static DeleteStatus skip_criteria(DeleteCondition cond, DeleteType typ, BBoxPtr<T> &bbox, const Eigen::Matrix<T, 3, 1> &center, T range);

    static DeleteStatus skip_criteria(DeleteCondition cond, DeleteType typ, MinMaxHolder<T> &bbox, const Eigen::Matrix<T, 3, 1> &center, T range);

    static bool point_check(Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &center, DeleteCondition cond, T range_sq);
};

// ................. SEARCH FUNCTION .................
template <typename T>
struct ComparePairs
{
    bool operator()(const std::pair<T, Eigen::Matrix<T, 3, 1>> &p1,
                    const std::pair<T, Eigen::Matrix<T, 3, 1>> &p2)
    {
        return p1.first < p2.first;
    }
};

template <typename T>
using SearchHeap = std::priority_queue<
    std::pair<T, Eigen::Matrix<T, 3, 1>>,
    std::vector<std::pair<T, Eigen::Matrix<T, 3, 1>>>,
    ComparePairs<T>>;
#endif