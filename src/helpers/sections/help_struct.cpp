#include "map_storage/sections/help_struct.hpp"

template <typename T>
void DeleteManager<T>::update(const Eigen::Matrix<T, 3, 1> &point)
{
    min = min.cwiseMin(point);
    max = max.cwiseMax(point);
}

template <typename T>
void DeleteManager<T>::update(const Eigen::Matrix<T, 3, 1> &point_a, const Eigen::Matrix<T, 3, 1> &point_b)
{
    update(point_a);
    update(point_b);
}

template <typename T>
DeleteStatus DeleteManager<T>::skip_criteria(DeleteCondition cond, DeleteType typ, BBoxPtr<T> &bbox, const Eigen::Matrix<T, 3, 1> &center, T range)
{
    auto oct_status = bbox->box_within_reference(center, range, typ);
    if (cond == DeleteCondition::Inside)
    {
        if (oct_status == BBox<T>::Status::Outside)
            return DeleteStatus::Collapse;

        if (oct_status == BBox<T>::Status::Inside)
            return DeleteStatus::Skip;

        return DeleteStatus::None;
    }

    // here we want to delete outside.. so reverse of above
    if (oct_status == BBox<T>::Status::Outside)
        return DeleteStatus::Skip;

    if (oct_status == BBox<T>::Status::Inside)
        return DeleteStatus::Collapse;

    return DeleteStatus::None;
}

template <typename T>
DeleteStatus DeleteManager<T>::skip_criteria(DeleteCondition cond, DeleteType typ, MinMaxHolder<T> &mm, const Eigen::Matrix<T, 3, 1> &center, T range)
{
    auto oct_status = BBox<T>::box_within_reference(center, mm.min, mm.max, range, typ);
    if (cond == DeleteCondition::Inside)
    {
        if (oct_status == BBox<T>::Status::Outside)
            return DeleteStatus::Collapse;

        if (oct_status == BBox<T>::Status::Inside)
            return DeleteStatus::Skip;

        return DeleteStatus::None;
    }

    // here we want to delete outside.. so reverse of above
    if (oct_status == BBox<T>::Status::Outside)
        return DeleteStatus::Skip;

    if (oct_status == BBox<T>::Status::Inside)
        return DeleteStatus::Collapse;

    return DeleteStatus::None;
}

template <typename T>
bool DeleteManager<T>::point_check(Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &center, DeleteCondition cond, T range_sq)
{
    if (cond == DeleteCondition::Inside)
        return ((point - center).squaredNorm() < range_sq);

    return ((point - center).squaredNorm() > range_sq);
}

template struct DeleteManager<double>;
template struct DeleteManager<float>;