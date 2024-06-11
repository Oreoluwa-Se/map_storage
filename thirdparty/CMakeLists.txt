# Installing thirdparty libraries if not currently existing

# run depenecy installe
function(install_dep)
    message(STATUS "Running dependency installer...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_SOURCE_DIR}/thirdparty ${CMAKE_SOURCE_DIR}/thirdparty/install_deps.sh
        RESULT_VARIABLE result
    )
    
    if (result)
        message(FATAL_ERROR "Failed to run dependncy installer")
    endif()
endfunction()

# ------------------- Eigen ------------------- 
find_package(Eigen3 REQUIRED)
if (NOT TARGET Eigen3::Eigen)
    message(STATUS "Eigen3 not found. Downloading and installing locally")
    install_dep()
endif()    

# ------------------- Find Boost ------------------- 
find_package(Boost QUIET)
if (NOT Boost_FOUND)
    message(STATUS "Boost not found. Downloading and installing locally")
    install_dep()
    find_package(Boost REQUIRED COMPONENTS thread system)
endif()


# ------------------- Find Sophus ------------------- 
find_package(Sophus QUIET)
if (NOT TARGET Sophus::Sophus)
    message(STATUS "Sophus not found. Downloading and installing locally")
    install_dep()
endif()

