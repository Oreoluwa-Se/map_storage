#ifndef BOUNDING_BOX_HPP
#define BOUNDING_BOX_HPP

#include "map_storage/utils/alias.hpp"
#include <atomic>
#include <boost/thread/shared_mutex.hpp>
#include <sstream>
#include <iomanip>

enum class DeleteCondition
{
    Inside,
    Outside
};

enum class DeleteType
{
    Box,
    Spherical,
};

template <typename T>
class BBox
{
public:
    using Ptr = std::shared_ptr<BBox<T>>;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class Status
    {
        Inside,     // point is strictly within the bounding box.
        Outside,    // point is outside the extended bounding box (considering the range).
        Borderline, // within the extended bounding box but closer to the boundary than the specified range
        Invalid
    };

    explicit BBox(bool _track_cov = false);

    explicit BBox(const Eigen::Matrix<T, 3, 1> &point, bool _track_cov = false);

    BBox(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, bool _track_cov = false);

    bool contains(const Eigen::Matrix<T, 3, 1> &point) const;

    bool intersects(const BBox<T> &other) const;

    bool intersects(const Ptr &other) const;

    Eigen::Matrix<T, 3, 1> get_max() const;

    Eigen::Matrix<T, 3, 1> get_mean() const;

    Eigen::Matrix<T, 3, 3> get_cov() const;

    Eigen::Matrix<T, 3, 1> get_min() const;

    Eigen::Matrix<T, 3, 1> center() const;

    void update(const Eigen::Matrix<T, 3, 1> &point);

    void min_max_update(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &pointb);

    void min_max_update(const Eigen::Matrix<T, 3, 1> &point);

    void set_min_max(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max);

    Status point_within_bbox(const Eigen::Matrix<T, 3, 1> &point, T range) const;

    static Status point_within_bbox(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, T range);

    Status box_within_reference(const Eigen::Matrix<T, 3, 1> &center, T range, DeleteType del_type);

    static Status box_within_reference(
        const Eigen::Matrix<T, 3, 1> &center, const Eigen::Matrix<T, 3, 1> &min,
        const Eigen::Matrix<T, 3, 1> &max, T range, DeleteType del_type);

    void update(const AVector3TVecCC<T> &points);

    void update(const Point3dPtrVectCC<T> &points);

    void update(const AVector3TVec<T> &points);

    void decrement(const AVector3TVecCC<T> &ptd);

    void decrement(const AVector3TVec<T> &ptd);

    void decrement(const Point3dPtrVectCC<T> &ptd);

    void update_size(size_t c_size);

    size_t get_size() const;

    std::string to_string() const;

    friend std::ostream &operator<<(std::ostream &os, const BBox<T> &bbox)
    {
        os << bbox.to_string();
        return os;
    }

    void unsafe_min_max_update(const Eigen::Matrix<T, 3, 1> &point);

    void unsafe_reset(bool min_max_only = false);

    void reset(bool min_max_only = false);

    static void status_print(Status stats);

    T closest_distance(const Eigen::Matrix<T, 3, 1> &point) const;

    static T closest_distance(const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max);

private: // function
    void init_mean_and_cov();

    void update_mean_and_cov(const Eigen::Matrix<T, 3, 1> &point);

    void solve_remove(const Eigen::Matrix<T, Eigen::Dynamic, 3> &points);

    static Status box_within_defined_sphere(
        const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min,
        const Eigen::Matrix<T, 3, 1> &max, T range);

    static Status box_within_defined_box(
        const Eigen::Matrix<T, 3, 1> &point, const Eigen::Matrix<T, 3, 1> &min,
        const Eigen::Matrix<T, 3, 1> &max, T range);

    void mean_cov_update(Eigen::Matrix<T, 3, 1> &batch_mean, Eigen::Matrix<T, 3, 3> &batch_cov, size_t batch_count);

private: // attributes
    Eigen::Matrix<T, 3, 1> min, max, mean;
    Eigen::Matrix<T, 3, 3> cov;
    AVector3TVec<T> temp_points;
    size_t num_points;
    bool track_cov = false;
    bool mean_cov_initialized = false;
    mutable boost::shared_mutex mutex;
};

// Type aliases
template <typename T>
using BBoxPtr = typename BBox<T>::Ptr;

template <typename T>
using BBoxStatus = typename BBox<T>::Status;

#endif