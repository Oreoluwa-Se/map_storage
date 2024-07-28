#include "map_storage/utils/point.hpp"
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

template <typename T>
Point3d<T>::Point3d()
    : point(Eigen::Matrix<T, 3, 1>::Zero()), intensity(0), timestamp(0.0), dt(0.0), octant_key(Point3d<T>::sign_cardinality(point)) {}

template <typename T>
Point3d<T>::Point3d(const Eigen::Matrix<T, 3, 1> &p, std::uint8_t intensity, T timestamp, T dt)
    : point(p), intensity(intensity), timestamp(timestamp), dt(dt), octant_key(Point3d<T>::sign_cardinality(p)) {}

template <typename T>
Point3d<T>::Point3d(T x, T y, T z, std::uint8_t intensity, T timestamp, T dt)
    : point(Eigen::Matrix<T, 3, 1>(x, y, z)), intensity(intensity), timestamp(timestamp), dt(dt), octant_key(Point3d<T>::sign_cardinality(point)) {}

template <typename T>
Point3d<T>::Point3d(const Point3d &other)
    : point(other.point), vox(other.vox), intensity(other.intensity), timestamp(other.timestamp), dt(other.dt),
      octant_key(other.octant_key), point_valid(other.point_valid), frame_id(other.frame_id) {}

template <typename T>
Point3d<T>::Point3d(Point3d &&other) noexcept
    : point(std::move(other.point)), vox(std::move(other.vox)), intensity(other.intensity), timestamp(other.timestamp), dt(other.dt),
      octant_key(other.octant_key), point_valid(other.point_valid), frame_id(other.frame_id) {}

template <typename T>
Point3d<T> &Point3d<T>::operator=(const Point3d<T> &other)
{
    if (this != &other)
    {
        point = other.point;
        vox = other.vox;
        octant_key = other.octant_key;
        intensity = other.intensity;
        timestamp = other.timestamp;
        frame_id = other.frame_id;
        point_valid = other.point_valid;
        dt = other.dt;
    }

    return *this;
}

template <typename T>
Point3d<T> &Point3d<T>::operator=(Point3d<T> &&other) noexcept
{
    if (this != &other)
    {
        point = std::move(other.point);
        vox = std::move(other.vox);
        octant_key = other.octant_key;
        intensity = other.intensity;
        timestamp = other.timestamp;
        frame_id = other.frame_id;
        point_valid = other.point_valid;
        dt = other.dt;
    }
    return *this;
}

template <typename T>
Eigen::Matrix<T, 3, 1> Point3d<T>::operator+(const Point3d<T> &other) const
{
    return point + other.point;
}

template <typename T>
Eigen::Matrix<T, 3, 1> Point3d<T>::operator-(const Point3d<T> &other) const
{
    return point - other.point;
}

template <typename T>
Eigen::Matrix<T, 3, 1> Point3d<T>::operator+(const Eigen::Matrix<T, 3, 1> &other) const
{
    return point + other;
}

template <typename T>
Eigen::Matrix<T, 3, 1> Point3d<T>::operator-(const Eigen::Matrix<T, 3, 1> &other) const
{
    return point - other;
}

template <typename T>
std::string Point3d<T>::to_string() const
{
    return "[" + std::to_string(point.x()) + ", " + std::to_string(point.y()) + ", " + std::to_string(point.z()) + "]";
}

template <typename T>
T Point3d<T>::sq_distance(const Point3d<T> &other) const { return (point - other.point).squaredNorm(); }

template <typename T>
T Point3d<T>::sq_distance(const Eigen::Matrix<T, 3, 1> &other) const { return (point - other).squaredNorm(); }

template <typename T>
T Point3d<T>::distance(const Point3d<T> &other) const { return (point - other.point).norm(); }

template <typename T>
T Point3d<T>::distance(const Eigen::Matrix<T, 3, 1> &other) const { return (point - other).norm(); }

template <typename T>
void Point3d<T>::print() const { std::cout << to_string() << std::endl; }

template <typename T>
T Point3d<T>::squaredNorm() const { return point.squaredNorm(); }

template <typename T>
T Point3d<T>::norm() const { return point.norm(); }

template <typename T>
T Point3d<T>::x() const { return point.x(); }

template <typename T>
T Point3d<T>::y() const { return point.y(); }

template <typename T>
T Point3d<T>::z() const { return point.z(); }

template <typename T>
void Point3d<T>::normalize() { point.normalize(); }

template <typename T>
int Point3d<T>::sign_cardinality(const Eigen::Matrix<T, 3, 1> &vec)
{
    // Initialize cardinality with 0
    int cardinality = 0;

    // Determine the sign of each element and add weighted values
    cardinality += (vec[0] >= 0 ? 1 : 0) * 1; // 2^0
    cardinality += (vec[1] >= 0 ? 1 : 0) * 2; // 2^1
    cardinality += (vec[2] >= 0 ? 1 : 0) * 4; // 2^2

    return cardinality;
}

template <typename T>
std::string Point3d<T>::eig_to_string(const Eigen::Matrix<T, 3, 1> &point)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "[" << point(0) << ", " << point(1) << ", " << point(2) << "]";
    return oss.str();
}

template <typename T>
std::string Point3d<T>::eig_to_string(const Eigen::Vector3i &point)
{
    std::ostringstream oss;
    oss << "[" << point(0) << ".0, " << point(1) << ".0, " << point(2) << ".0]";
    return oss.str();
}

template <typename T>
Eigen::Vector3i Point3d<T>::calc_vox_index(const Point3d<T>::Ptr &point, T vox_size)
{

    // Scale the adj coordinates to voxel grid.
    T inv_vox = T(1.0) / vox_size;
    return Eigen::Vector3i(
        static_cast<int>(point->x() * inv_vox),
        static_cast<int>(point->y() * inv_vox),
        static_cast<int>(point->z() * inv_vox));
}

template <typename T>
Eigen::Vector3i Point3d<T>::calc_vox_index(const Eigen::Matrix<T, 3, 1> &point, T vox_size)
{
    // Scale the adj coordinates to voxel grid.
    T inv_vox = T(1.0) / vox_size;
    return Eigen::Vector3i(
        static_cast<int>(point.x() * inv_vox),
        static_cast<int>(point.y() * inv_vox),
        static_cast<int>(point.z() * inv_vox));
}

template struct Point3d<double>;
template struct Point3d<float>;
