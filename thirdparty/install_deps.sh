#!/bin/bash

# exit immediately if a command exits witha non-zero staus
set -e

# Install Eigen3
if ! cmake --find-package -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST -DNAME=Eigen3; then
    echo "Installing Eigen3..."
    git clone https://gitlab.com/libeigen/eigen.git
    cd eigen
    mkdir build && cd build
    cmake ..
    sudo make install
    cd ../..
    rm -rf eigen
fi

# Install Boost
if ! cmake --find-package -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST -DNAME=Boost; then
    echo "Installing Boost..."
    git clone --recursive https://github.com/boostorg/boost.git
    cd boost
    ./bootstrap.sh
    sudo ./b2 install
    cd ..
    rm -rf boost
fi

# Install TBB
if ! cmake --find-package -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST -DNAME=TBB; then
    echo "Installing TBB..."
    git clone https://github.com/oneapi-src/oneTBB.git
    cd oneTBB
    mkdir build && cd build
    cmake ..
    sudo make install
    cd ../..
    rm -rf oneTBB
fi

# Install Sophus
if ! cmake --find-package -DCOMPILER_ID=GNU -DLANGUAGE=CXX -DMODE=EXIST -DNAME=Sophus; then
    echo "Installing Sophus..."
    git clone https://github.com/strasdat/Sophus.git
    cd Sophus
    mkdir build && cd build
    cmake ..
    sudo make install
    cd ../..
    rm -rf Sophus
fi