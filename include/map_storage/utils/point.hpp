#ifndef BASE_3D_POINTS_HPP
#define BASE_3D_POINTS_HPP

#include <Eigen/Dense>
#include <memory>
#include <iostream>

template <typename T>
struct Point3d
{
    using Ptr = std::shared_ptr<Point3d<T>>;
    using WPtr = std::weak_ptr<Point3d<T>>;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Point3d();

    explicit Point3d(const Eigen::Matrix<T, 3, 1> &p, std::uint8_t intensity = 0, T timestamp = 0.0, T dt = 0.0);

    Point3d(T x, T y, T z, std::uint8_t intensity = 0, T timestamp = 0.0, T dt = 0.0);

    Point3d(const Point3d &other);

    Point3d(Point3d &&other) noexcept;

    Point3d<T> &operator=(const Point3d<T> &other);

    Point3d<T> &operator=(Point3d<T> &&other) noexcept;

    Eigen::Matrix<T, 3, 1> operator+(const Point3d<T> &other) const;

    Eigen::Matrix<T, 3, 1> operator-(const Point3d<T> &other) const;

    Eigen::Matrix<T, 3, 1> operator+(const Eigen::Matrix<T, 3, 1> &other) const;

    Eigen::Matrix<T, 3, 1> operator-(const Eigen::Matrix<T, 3, 1> &other) const;

    std::string to_string() const;

    void print() const;

    void normalize();

    T sq_distance(const Point3d<T> &other) const;

    T sq_distance(const Eigen::Matrix<T, 3, 1> &other) const;

    T distance(const Point3d<T> &other) const;

    T distance(const Eigen::Matrix<T, 3, 1> &other) const;

    T squaredNorm() const;

    T norm() const;

    T x() const;

    T y() const;

    T z() const;

    static int sign_cardinality(const Eigen::Matrix<T, 3, 1> &vec);

    static std::string eig_to_string(const Eigen::Matrix<T, 3, 1> &point);

    static std::string eig_to_string(const Eigen::Vector3i &point);

    static Eigen::Vector3i calc_vox_index(const Ptr &point, T vox_size);

    static Eigen::Vector3i calc_vox_index(const Eigen::Matrix<T, 3, 1> &point, T vox_size);

public:                                                            // attributes
    Eigen::Matrix<T, 3, 1> point = Eigen::Matrix<T, 3, 1>::Zero(); // 3d point in the map
    Eigen::Vector3i vox;                                           // voxel representation
    std::uint8_t intensity = 0;                                    // intensity value
    T timestamp = T(0.0), dt = T(0.0);                             // timestamp and dt value
    int octant_key = 0;                                            // used for point registration
    bool point_valid = false;
    int frame_id = -1;
};

// Type aliases
template <typename T>
using Point3dPtr = typename Point3d<T>::Ptr;

template <typename T>
using Point3dWPtr = typename Point3d<T>::WPtr;

#endif