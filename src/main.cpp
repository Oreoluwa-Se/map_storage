#include "run.hpp"

int main(int argc, char **argv)
{
    size_t N = 2000;
    double range = 5.0;
    double delete_range = 1.0;
    bool verbose = true;
    size_t num_nearest = 5;
    bool track_cov = true; // the voxels will track its mean and covariance values

    RunFunctions<double> opt;
    std::cout << "Testing insert scheme options" << std::endl;
    opt.testing_insert_schemes(N, range, track_cov);

    size_t t_range = range;
    std::cout << "\nTesting the search scheme" << std::endl;
    opt.testing_search(N, t_range, track_cov, num_nearest, verbose);

    std::cout << "\nTesting the delete scheme" << std::endl;
    opt.testing_combined_delete(N, range, delete_range, verbose);

    std::cout << "\nDoing a test based on the faster lio paper" << std::endl;
    opt.incremental_info(range, 100);

    return 0;
}