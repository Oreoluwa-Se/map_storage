#ifndef MAP_LOADER_HPP
#define MAP_LOADER_HPP

#include <yaml-cpp/yaml.h>
#include <iostream>

struct ReadParams
{
    static void run(const std::string &loc);
};

extern YAML::Node test_params;
extern YAML::Node build_params;

template <typename T>
inline void set_param(T &param, const YAML::Node &node, const std::string &frm = " ")
{
    if (node.IsDefined() && !node.IsNull())
    {
        try
        {
            param = node.as<T>();
        }
        catch (const YAML::TypedBadConversion<T> &e)
        {
            std::cerr << "YAML conversion error from " << frm << " : " << e.what() << std::endl;
        }
    }
    else
        std::cerr << "YAML node is undefined or null when extracting " << frm << "." << std::endl;
}

#endif