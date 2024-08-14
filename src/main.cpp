#include "run.hpp"
#include <string>
#include "map_storage/utils/loader.hpp"

int main(int argc, char **argv)
{
    // config file is relative to run directory.
    std::string config_loc = "../../config/params.yaml";
    ReadParams::run(config_loc);

    RunFunctions<double> opt(true);

    bool test_val;
    set_param(test_val, test_params["testing_insert_search_run"]);
    if (test_val)
        opt.testing_incremental();

    // running faster lio test
    set_param(test_val, test_params["faster_lio_test"]);
    if (test_val)
    {
        std::cout << "\nDoing a test based on the faster lio paper" << std::endl;
        opt.incremental_info();
    }

    // basic test
    set_param(test_val, test_params["basic_test"]);
    if (test_val)
    {
        opt.testing_insert_schemes();
        opt.testing_search();
        opt.testing_downsample_scheme();
        opt.testing_combined_delete();
        opt.test_point_retrival();
    }

    return 0;
}