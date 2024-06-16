#ifndef ALIAS_SETUP_HPP
#define ALIAS_SETUP_HPP

#include "point.hpp"
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <type_traits>
#include <vector>

// ...................... CREATING VECTOR ALIASES .............................
template <typename T>
using Point3dPtrVect = std::vector<Point3dPtr<T>>;

template <typename T>
using Point3dWPtrVec = std::vector<Point3dWPtr<T>>;

template <typename T>
using Point3dPtrVectCC = tbb::concurrent_vector<Point3dPtr<T>>;

template <typename T>
using Point3dPtrCCQueue = tbb::concurrent_queue<Point3dPtr<T>>;

template <typename T>
using Point3dWPtrCCQueue = tbb::concurrent_queue<Point3dWPtr<T>>;

template <typename T>
using Point3dWPtrVecCC = tbb::concurrent_vector<Point3dWPtr<T>>;

template <typename T>
using VisualizationPointStorage = tbb::concurrent_vector<Point3dWPtrVecCC<T>>;

template <typename T>
using APointVector = std::vector<Point3d<T>, Eigen::aligned_allocator<Point3d<T>>>;

using AVector3iVector = std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>>;

template <typename T>
using AVector3TVec = std::vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>>;

template <typename T>
using AVector4TVec = std::vector<Eigen::Matrix<T, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 4, 1>>>;

template <typename T>
using AVector3TVecCC = tbb::concurrent_vector<Eigen::Matrix<T, 3, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 1>>>;

template <typename T>
using AVector4TVecCC = tbb::concurrent_vector<Eigen::Matrix<T, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<T, 4, 1>>>;

template <typename T>
using A3x3T = std::vector<Eigen::Matrix<T, 3, 3>, Eigen::aligned_allocator<Eigen::Matrix<T, 3, 3>>>;

template <typename T>
using TMatType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using DistancePointPair = std::pair<T, Point3dPtr<T>>;

// ...................... CREATING SOPHUS TEMPLATE .............................
// Primary template
template <typename T, typename Enable = void>
struct SophusType;

// Specialization for double
template <typename T>
struct SophusType<T, typename std::enable_if<std::is_same<T, double>::value>::type>
{
    using SE3 = Sophus::SE3d;
    using SO3 = Sophus::SO3d;
};

// Specialization for float
template <typename T>
struct SophusType<T, typename std::enable_if<std::is_same<T, float>::value>::type>
{
    using SE3 = Sophus::SE3f;
    using SO3 = Sophus::SO3f;
};

template <typename T>
using SE3Type = typename SophusType<T>::SE3;

// Alias template for SO3
template <typename T>
using SO3Type = typename SophusType<T>::SO3;

#endif