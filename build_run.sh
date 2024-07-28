#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 {debug|release}"
    exit 1
}

# Check if the correct number of arguments is provided
if [ $# -ne 1 ]; then
    usage
fi

# Determine the mode based on the argument
MODE=$1
BUILD_DIR=""
BUILD_TYPE=""
LOG_FILE=""

if [ "$MODE" == "debug" ]; then
    BUILD_DIR="build/debug"
    BUILD_TYPE="Debug"
    LOG_FILE="debug_log.txt"
elif [ "$MODE" == "release" ]; then
    BUILD_DIR="build/release"
    BUILD_TYPE="Release"
    LOG_FILE="release_log.txt"
else
    usage
fi

# Create build directory and change to it
mkdir -p $BUILD_DIR && cd $BUILD_DIR || { echo "Failed to create or navigate to build directory"; exit 1; }

# Run CMake
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ../.. || { echo "CMake configuration failed"; exit 1; }

# Run Make
make || { echo "Make failed"; exit 1; }

# Clear the terminal
clear

# Build the project
cmake --build . || { echo "Build failed"; exit 1; }

# Run the executable and direct output to log file
if [ "$MODE" == "debug" ]; then
    # gdb ./bin/map_storage_exec
    ./bin/map_storage_exec > $LOG_FILE 2>&1 || { echo "Execution failed"; exit 1; }
    echo "Output if any are displayed in $BUILD_DIR/$LOG_FILE"
else
    ./bin/map_storage_exec > $LOG_FILE 2>&1 || { echo "Execution failed"; exit 1; }
    echo "Output if any are displayed in $BUILD_DIR/$LOG_FILE"
fi
