#ifndef MAP_TRIMMER_HPP
#define MAP_TRIMMER_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/utils/alias.hpp"
#include "maphub.hpp"
#include <memory>

template <typename T>
struct Deleter : public std::enable_shared_from_this<Deleter<T>>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr = std::shared_ptr<Deleter<T>>;

    void set_support_info(ConfigPtr<T> &config_ptr);

    void delete_within_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type = DeleteType::Spherical);

    void delete_outside_points(const Eigen::Matrix<T, 3, 1> &ptd, T range, DeleteType del_type = DeleteType::Spherical);

    void range_delete(BlockPtr<T> &c_block, const Eigen::Matrix<T, 3, 1> &center, T range, DeleteCondition cond, DeleteType del_type);

private:
    void handle_collapse(BlockPtr<T> c_block, DeleteCondition cond);

private:
    ConfigPtr<T> config = nullptr;
};

template <typename T>
using DeleterPtr = typename Deleter<T>::Ptr;
#endif