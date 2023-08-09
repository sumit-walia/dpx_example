#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include "dpx.cuh"

// For parsing the command line values
namespace po = boost::program_options;


int main(int argc, char** argv) {


    // Print GPU information
    // printGpuProperties();
    
    dpx::smith_waterman();
    fprintf(stdout, "Completed\n\n");

    return 0;
}
