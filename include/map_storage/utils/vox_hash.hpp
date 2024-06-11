#ifndef VOXELHASH_HPP
#define VOXELHASH_HPP

#include <Eigen/Dense>
#include <cstddef>

struct VoxelHash
{
    static size_t hash(const Eigen::Vector3i &voxel)
    {
        std::size_t seed = 0;
        seed ^= std::hash<int>{}(voxel[0]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(voxel[1]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>{}(voxel[2]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }

    size_t operator()(const Eigen::Vector3i &voxel) const
    {
        return hash(voxel);
    }

    static bool equal(const Eigen::Vector3i &a, const Eigen::Vector3i &b)
    {
        return a == b;
    }
};

struct VoxelHashEqual
{
    bool operator()(const Eigen::Vector3i &a, const Eigen::Vector3i &b) const
    {
        return a == b;
    }
};

#endif