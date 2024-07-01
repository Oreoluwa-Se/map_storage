#include "map_storage/utils/bbox.hpp"
#include <limits>

template <typename T>
BBox<T>::BBox(bool _track_cov)
    : min(Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::max()),
      max(Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::lowest()),
      mean(Eigen::Matrix<T, 3, 1>::Zero()),
      cov(Eigen::Matrix<T, 3, 3>::Identity() * 1e9),
      num_points(0), track_cov(_track_cov) {}

template <typename T>
BBox<T>::BBox(const Eigen::Matrix<T, 3, 1> &point, bool _track_cov)
    : min(point), max(point), mean(Eigen::Matrix<T, 3, 1>::Zero()),
      cov(Eigen::Matrix<T, 3, 3>::Identity() * 1e9),
      num_points(0), track_cov(_track_cov) {}

template <typename T>
BBox<T>::BBox(const Eigen::Matrix<T, 3, 1> &min_point, const Eigen::Matrix<T, 3, 1> &max_point, bool _track_cov)
    : min(min_point), max(min_point), mean(Eigen::Matrix<T, 3, 1>::Zero()),
      cov(Eigen::Matrix<T, 3, 3>::Identity() * 1e9),
      num_points(0), track_cov(_track_cov)
{
    min = min.cwiseMin(max_point);
    max = max.cwiseMax(max_point);
}

template <typename T>
Eigen::Matrix<T, 3, 1> BBox<T>::get_max() const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return max;
}

template <typename T>
Eigen::Matrix<T, 3, 1> BBox<T>::get_mean() const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return mean;
}

template <typename T>
Eigen::Matrix<T, 3, 3> BBox<T>::get_cov() const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return cov;
}

template <typename T>
Eigen::Matrix<T, 3, 1> BBox<T>::get_min() const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return min;
}

template <typename T>
size_t BBox<T>::get_size() const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return num_points;
}

template <typename T>
void BBox<T>::update_size(size_t c_size)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    num_points -= c_size;
}

template <typename T>
bool BBox<T>::contains(const Eigen::Matrix<T, 3, 1> &point) const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return (point[0] >= min[0] && point[0] <= max[0] &&
            point[1] >= min[1] && point[1] <= max[1] &&
            point[2] >= min[2] && point[2] <= max[2]);
}

template <typename T>
bool BBox<T>::intersects(const BBox<T> &other) const
{
    Eigen::Matrix<T, 3, 1> o_min = other.get_min();
    Eigen::Matrix<T, 3, 1> o_max = other.get_max();

    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return (min[0] <= o_max[0] && max[0] >= o_min[0] &&
            min[1] <= o_max[1] && max[1] >= o_min[1] &&
            min[2] <= o_max[2] && max[2] >= o_min[2]);
}

template <typename T>
bool BBox<T>::intersects(const typename BBox<T>::Ptr &other) const
{
    Eigen::Matrix<T, 3, 1> o_min = other->get_min();
    Eigen::Matrix<T, 3, 1> o_max = other->get_max();

    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return (min[0] <= o_max[0] && max[0] >= o_min[0] &&
            min[1] <= o_max[1] && max[1] >= o_min[1] &&
            min[2] <= o_max[2] && max[2] >= o_min[2]);
}

template <typename T>
Eigen::Matrix<T, 3, 1> BBox<T>::center() const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return (min + max) * 0.5;
}

template <typename T>
void BBox<T>::set_min_max(const Eigen::Matrix<T, 3, 1> &min_, const Eigen::Matrix<T, 3, 1> &max_)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    min = min_;
    max = max_;
}

template <typename T>
void BBox<T>::min_max_update(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &pointb)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);

    min = min.cwiseMin(point);
    min = min.cwiseMin(pointb);

    max = max.cwiseMax(point);
    max = max.cwiseMax(pointb);
}

template <typename T>
void BBox<T>::min_max_update(const Eigen::Matrix<T, 3, 1> &point)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    min = min.cwiseMin(point);
    max = max.cwiseMax(point);
}

template <typename T>
void BBox<T>::status_print(Status stats)
{
    switch (stats)
    {
    case Status::Inside:
        std::cout << "Inside" << std::endl;
        break;
    case Status::Outside:
        std::cout << "Outside" << std::endl;
        break;
    case Status::Borderline:
        std::cout << "Borderline" << std::endl;
        break;
    case Status::Invalid:
        std::cout << "Invalid" << std::endl;
        break;
    }
}

template <typename T>
typename BBox<T>::Status BBox<T>::point_within_bbox(const Eigen::Matrix<T, 3, 1> &point, T range) const
{

    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return BBox<T>::point_within_bbox(point, min, max, range);
}

template <typename T>
typename BBox<T>::Status BBox<T>::point_within_bbox(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, T range)
{
    if (range < T(0))
        return Status::Invalid;

    if ((point.array() >= min.array()).all() && (point.array() <= max.array()).all())
        return Status::Inside;

    Eigen::Matrix<T, 3, 1> min_b = min - Eigen::Matrix<T, 3, 1>::Constant(range);
    Eigen::Matrix<T, 3, 1> max_b = max + Eigen::Matrix<T, 3, 1>::Constant(range);
    if ((point.array() < min_b.array()).any() || (point.array() > max_b.array()).any())
        return Status::Outside;

    return Status::Borderline;
}

template <typename T>
typename BBox<T>::Status BBox<T>::box_within_reference(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type)
{

    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return BBox<T>::box_within_reference(center, min, max, range, del_type);
}

template <typename T>
typename BBox<T>::Status BBox<T>::box_within_reference(
    const Eigen::Matrix<T, 3, 1> &center, const Eigen::Matrix<T, 3, 1> &min,
    const Eigen::Matrix<T, 3, 1> &max, T range, DeleteType del_type)
{
    // status check
    if (del_type == DeleteType::Spherical)
        return BBox<T>::box_within_defined_sphere(center, min, max, range);

    if (del_type == DeleteType::Box)
        return BBox<T>::box_within_defined_box(center, min, max, range);

    return BBox<T>::Status::Borderline;
}

template <typename T>
typename BBox<T>::Status BBox<T>::box_within_defined_sphere(
    const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min,
    const Eigen::Matrix<T, 3, 1> &max, T range)
{
    // checks if the point is within sphere created by point and range
    if (range < 0)
        return Status::Invalid;

    // Compute the centroid of the bounding box
    Eigen::Matrix<T, 3, 1> centroid = 0.5 * (min + max);

    // Compute the half-diagonal of the bounding box
    T half_diag = (max - min).norm() / 2.0;

    T dist_to_centroid = (point - centroid).norm();
    if (dist_to_centroid + half_diag <= range)
        return Status::Inside;

    // Check if any part of the bounding box is inside the sphere
    Eigen::Matrix<T, 3, 1> c_point = point.cwiseMax(min).cwiseMin(max);
    if ((point - c_point).squaredNorm() <= range * range)
        return Status::Borderline;

    return Status::Outside;
}

template <typename T>
typename BBox<T>::Status BBox<T>::box_within_defined_box(
    const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min,
    const Eigen::Matrix<T, 3, 1> &max, T range)
{
    // Check for invalid input
    if (range < 0)
        return Status::Invalid;

    // Compute the limits of the enclosing box centered at the given point with the specified range
    Eigen::Matrix<T, 3, 1> enc_min = point - Eigen::Matrix<T, 3, 1>::Constant(range);
    Eigen::Matrix<T, 3, 1> enc_max = point + Eigen::Matrix<T, 3, 1>::Constant(range);

    // Check if the bounding box is entirely within the enclosing box
    if ((min.array() >= enc_min.array()).all() && (max.array() <= enc_max.array()).all())
        return Status::Inside;

    // Check if the bounding box is entirely outside the enclosing box
    if ((max.array() < enc_min.array()).any() || (min.array() > enc_max.array()).any())
        return Status::Outside;

    return Status::Borderline;
}

template <typename T>
void BBox<T>::unsafe_min_max_update(const Eigen::Matrix<T, 3, 1> &point)
{

    min = min.cwiseMin(point);
    max = max.cwiseMax(point);

    if (track_cov)
    {
        if (!mean_cov_initialized)
        {
            ++num_points;
            temp_points.push_back(point);
            if (num_points >= 3)
                init_mean_and_cov();
        }
        else
            update_mean_and_cov(point);
    }
    else
        ++num_points;
}
template <typename T>
void BBox<T>::update(const Eigen::Matrix<T, 3, 1> &point)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    unsafe_min_max_update(point);
}

template <typename T>
void BBox<T>::init_mean_and_cov()
{
    mean = Eigen::Matrix<T, 3, 1>::Zero();
    for (const auto &p : temp_points)
        mean.noalias() += p;

    mean /= static_cast<T>(temp_points.size());

    cov = Eigen::Matrix<T, 3, 3>::Zero();
    for (const auto &p : temp_points)
    {
        Eigen::Matrix<T, 3, 1> diff = p - mean;
        cov.noalias() += diff * diff.transpose();
    }
    cov /= static_cast<T>(temp_points.size() - 1);

    mean_cov_initialized = true;
    // clearing information
    temp_points.clear();
}

template <typename T>
void BBox<T>::update_mean_and_cov(const Eigen::Matrix<T, 3, 1> &point)
{
    ++num_points;
    Eigen::Matrix<T, 3, 1> prev_mean = mean;
    T num = static_cast<T>(num_points);

    // Update mean incrementally
    mean = (mean * (num - 1.0) + point) / num;

    // Update cov incrementally
    Eigen::Matrix<T, 3, 1> old_diff = point - prev_mean;
    Eigen::Matrix<T, 3, 1> new_diff = point - mean;

    cov = ((num - 2.0) * cov + old_diff * new_diff.transpose()) / (num - 1.0);
}

template <typename T>
void BBox<T>::reset(bool min_max_only)
{
    boost::unique_lock<boost::shared_mutex> lock(mutex);
    unsafe_reset(min_max_only);
}

template <typename T>
void BBox<T>::unsafe_reset(bool min_max_only)
{
    min = Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::max();
    max = Eigen::Matrix<T, 3, 1>::Identity() * std::numeric_limits<T>::lowest();
    if (!min_max_only)
    {
        num_points = 0;
        mean_cov_initialized = false;
    }
}

template <typename T>
void BBox<T>::mean_cov_update(Eigen::Matrix<T, 3, 1> &batch_mean, Eigen::Matrix<T, 3, 3> &batch_cov, size_t batch_count)
{
    if (!mean_cov_initialized)
    {
        mean = batch_mean;
        cov = batch_cov;
        mean_cov_initialized = true;
    }
    else
    {
        T tot_points = static_cast<T>(num_points + batch_count);
        Eigen::Matrix<T, 3, 1> new_mean = (mean * num_points + batch_mean * batch_count) / tot_points;

        Eigen::Matrix<T, 3, 3> term1 = (num_points - 1) * cov;
        Eigen::Matrix<T, 3, 3> term2 = (batch_count - 1) * batch_cov;
        Eigen::Matrix<T, 3, 3> term3 = num_points * batch_count * (mean - batch_mean) * (mean - batch_mean).transpose() / tot_points;

        cov = (term1 + term2 + term3) / (tot_points - 1.0);
        mean = new_mean;
    }
}

template <typename T>
void BBox<T>::update(const AVector3TVecCC<T> &points)
{
    size_t batch_count = points.size();
    if (batch_count == 0)
        return;

    if (points.size() == 1)
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        unsafe_min_max_update(points[0]);
        return;
    }

    // Compute batch mean and cov
    Eigen::Matrix<T, 3, 1> batch_mean = Eigen::Matrix<T, 3, 1>::Zero();
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        for (const auto &point : points)
        {
            batch_mean.noalias() += point;
            min = min.cwiseMin(point);
            max = max.cwiseMax(point);
        }
    }

    if (track_cov)
    {
        batch_mean /= static_cast<T>(batch_count);

        Eigen::Matrix<T, 3, 3> batch_cov = Eigen::Matrix<T, 3, 3>::Zero();
        for (const auto &point : points)
        {
            Eigen::Matrix<T, 3, 1> diff = point - batch_mean;
            batch_cov.noalias() += diff * diff.transpose();
        }
        batch_cov /= static_cast<T>(batch_count - 1);

        boost::unique_lock<boost::shared_mutex> lock(mutex);
        mean_cov_update(batch_mean, batch_cov, batch_count);
    }

    num_points += batch_count;
}

template <typename T>
void BBox<T>::update(const Point3dPtrVectCC<T> &points)
{
    size_t batch_count = points.size();
    if (batch_count == 0)
        return;

    if (points.size() == 1)
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        unsafe_min_max_update(points[0]->point);
        return;
    }

    // Compute batch mean and cov
    Eigen::Matrix<T, 3, 1> batch_mean = Eigen::Matrix<T, 3, 1>::Zero();
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        for (const auto &point : points)
        {
            batch_mean.noalias() += point->point;
            min = min.cwiseMin(point->point);
            max = max.cwiseMax(point->point);
        }
    }

    if (track_cov)
    {
        batch_mean /= static_cast<T>(batch_count);

        Eigen::Matrix<T, 3, 3> batch_cov = Eigen::Matrix<T, 3, 3>::Zero();
        for (const auto &point : points)
        {
            Eigen::Matrix<T, 3, 1> diff = point->point - batch_mean;
            batch_cov.noalias() += diff * diff.transpose();
        }
        batch_cov /= static_cast<T>(batch_count - 1);

        boost::unique_lock<boost::shared_mutex> lock(mutex);
        mean_cov_update(batch_mean, batch_cov, batch_count);
    }

    num_points += batch_count;
}

template <typename T>
void BBox<T>::update(const AVector3TVec<T> &points)
{
    size_t batch_count = points.size();
    if (batch_count == 0)
        return;

    if (points.size() == 1)
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        unsafe_min_max_update(points[0]);
        return;
    }

    // Compute batch mean and cov
    Eigen::Matrix<T, 3, 1> batch_mean = Eigen::Matrix<T, 3, 1>::Zero();
    Eigen::Matrix<T, 3, 3> batch_cov = Eigen::Matrix<T, 3, 3>::Zero();
    {
        boost::unique_lock<boost::shared_mutex> lock(mutex);
        for (const auto &point : points)
        {
            batch_mean += point;
            min = min.cwiseMin(point);
            max = max.cwiseMax(point);
        }
    }

    if (track_cov)
    {
        batch_mean /= static_cast<T>(batch_count);
        for (const auto &point : points)
        {
            Eigen::Matrix<T, 3, 1> diff = point - batch_mean;
            batch_cov += diff * diff.transpose();
        }
        batch_cov /= static_cast<T>(batch_count - 1);

        boost::unique_lock<boost::shared_mutex> lock(mutex);
        mean_cov_update(batch_mean, batch_cov, batch_count);
    }

    num_points += batch_count;
}

template <typename T>
void BBox<T>::solve_remove(const Eigen::Matrix<T, Eigen::Dynamic, 3> &points)
{
    // batch_size
    const size_t batch_size = points.rows();
    if (batch_size == 0)
        return;

    boost::unique_lock<boost::shared_mutex> lock(mutex);

    if (track_cov)
    {
        T total_size = static_cast<T>(num_points - batch_size);
        // if we are removing everything
        if (num_points == batch_size || total_size <= T(0))
        {
            cov = Eigen::Matrix<T, 3, 3>::Constant(1e9);
            num_points = 0;
            mean.setZero();
            return;
        }

        T t_curr_points = static_cast<T>(num_points);
        T t_batch_size = static_cast<T>(batch_size);
        Eigen::Matrix<T, 3, 1> c_mean = points.topRows(batch_size).colwise().mean();
        Eigen::Matrix<T, 3, 1> updated_mean = (mean * t_curr_points - c_mean * t_batch_size) / total_size;

        if ((num_points - batch_size) < 3)
        { // we don't have sufficient points to calculate a covariance.
            cov.setConstant(1e9);
        }
        else if (batch_size == 1)
        {
            Eigen::Matrix<T, 3, 1> point_diff = c_mean - mean;
            Eigen::Matrix<T, 3, 3> rank_one_update = point_diff * point_diff.transpose();
            cov = ((cov * (t_curr_points - 1)) - rank_one_update) / (total_size - 1);
        }
        else
        {
            // All other updates to covariance
            Eigen::Matrix<T, Eigen::Dynamic, 3> zero_mean_mat = points.topRows(batch_size).rowwise() - mean.transpose();
            Eigen::Matrix<T, 3, 3> c_cov = (zero_mean_mat.transpose() * zero_mean_mat) / static_cast<T>(batch_size - 1);

            Eigen::Matrix<T, 3, 3> left_s = cov * (t_curr_points - 1.0) - (c_cov * (t_batch_size - 1.0)) / (total_size - 1.0);
            Eigen::Matrix<T, 3, 3> right_s = ((t_curr_points * t_batch_size) / (total_size * (total_size - 1.0))) * (mean - c_mean) * (mean - c_mean).transpose();

            cov = left_s - right_s;
        }

        // update number of points
        mean = updated_mean;
    }

    num_points -= batch_size;
}

template <typename T>
void BBox<T>::decrement(const AVector3TVecCC<T> &ptd)
{
    if (ptd.empty())
        return;

    Eigen::Matrix<T, Eigen::Dynamic, 3> del_points = Eigen::Matrix<T, Eigen::Dynamic, 3>::Zero(ptd.size(), 3);
    for (size_t idx = 0; idx < ptd.size(); ++idx)
        del_points.row(idx) = ptd[idx];

    solve_remove(del_points);
}

template <typename T>
void BBox<T>::decrement(const AVector3TVec<T> &ptd)
{
    if (ptd.empty())
        return;

    Eigen::Matrix<T, Eigen::Dynamic, 3> del_points = Eigen::Matrix<T, Eigen::Dynamic, 3>::Zero(ptd.size(), 3);
    for (size_t idx = 0; idx < ptd.size(); ++idx)
        del_points.row(idx) = ptd[idx];

    solve_remove(del_points);
}

template <typename T>
void BBox<T>::decrement(const Point3dPtrVectCC<T> &ptd)
{
    // Pointptr3d vector
    if (ptd.empty())
        return;

    Eigen::Matrix<T, Eigen::Dynamic, 3> del_points = Eigen::Matrix<T, Eigen::Dynamic, 3>::Zero(ptd.size(), 3);
    for (size_t idx = 0; idx < ptd.size(); ++idx)
        del_points.row(idx) = ptd[idx]->point;

    solve_remove(del_points);
}

template <typename T>
std::string BBox<T>::to_string() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3); // Set fixed-point notation and precision

    oss << "BoundingBox Information:" << std::endl;
    oss << "-------------------------" << std::endl;
    oss << "Min: [" << min(0) << ", " << min(1) << ", " << min(2) << "]" << std::endl;
    oss << "Max: [" << max(0) << ", " << max(1) << ", " << max(2) << "]" << std::endl;
    oss << "Mean: [" << mean(0) << ", " << mean(1) << ", " << mean(2) << "]" << std::endl;
    oss << "Covariance Matrix: " << std::endl;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            oss << std::setw(10) << cov(i, j) << " ";
        }
        oss << std::endl;
    }

    oss << "Number of Points: " << num_points << std::endl;

    return oss.str();
}

template <typename T>
T BBox<T>::closest_distance(const Eigen::Matrix<T, 3, 1> &point) const
{
    boost::shared_lock<boost::shared_mutex> lock(mutex);
    return BBox<T>::closest_distance(point, min, max);
}

template <typename T>
T BBox<T>::closest_distance(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max)
{
    Eigen::Matrix<T, 3, 1> clamped = point.cwiseMin(max).cwiseMax(min);
    return (clamped - point).squaredNorm();
}

template class BBox<double>;
template class BBox<float>;