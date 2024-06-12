#include "map_storage/utils/loader.hpp"

YAML::Node test_params;
YAML::Node build_params;

void ReadParams::run(const std::string &loc)
{
    YAML::Node node = YAML::LoadFile(loc);
    test_params = node["Testing"];
    build_params = node["Build"];
}