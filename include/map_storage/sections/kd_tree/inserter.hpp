#ifndef MAP_INSERTER_HPP
#define MAP_INSERTER_HPP

#include "map_storage/sections/kd_tree/node.hpp"
#include "map_storage/utils/alias.hpp"
#include "maphub.hpp"

template <typename T>
struct Inserter
{
    using Ptr = std::shared_ptr<Inserter<T>>;

    void set_support_info(ConfigPtr<T> &config_ptr);

    void insert(Point3dPtrVect<T> &points, BlockPtrVecCC<T> &scapegoats);

    void insert(Point3dPtrVectCC<T> &points, BlockPtrVecCC<T> &scapegoats);

    void insertion_cont(BlockPtr<T> &start_point, BlockPtr<T> &block, RunningStats<T> &r_stats);

private:
    void insertion_handler(BlockPtr<T> &start_point, BlockPtr<T> &block, BlockPtrVecCC<T> &scapegoats);

private:
    ConfigPtr<T> config = nullptr;
};

template <typename T>
using InserterPtr = typename Inserter<T>::Ptr;
#endif
