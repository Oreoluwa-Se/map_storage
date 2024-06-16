#include "run.hpp"
#include <string>
#include "map_storage/utils/loader.hpp"

int main(int argc, char **argv)
{
    // config file is relative to run directory.
    std::string config_loc = "../../config/params.yaml";
    ReadParams::run(config_loc);

    RunFunctions<double> opt(true);

    opt.testing_insert_schemes();
    opt.testing_search();
    opt.testing_downsample_scheme();
    opt.testing_combined_delete();
    opt.test_point_retrival();

    bool faster_lio_test;
    set_param(faster_lio_test, test_params["faster_lio_test"]);
    if (faster_lio_test)
    {
        std::cout << "\nDoing a test based on the faster lio paper" << std::endl;
        opt.incremental_info();
    }

    return 0;
}