cmake_minimum_required(VERSION 3.10)
project(map_storage)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# include directories 
include_directories(include)

# add dependencies
add_subdirectory(thirdparty)

# Source files for the library
file(GLOB_RECURSE LIB_SOURCES "src/helpers/*.cpp")
file(GLOB_RECURSE LIB_HEADERS "include/*.hpp" "include/*.h")

# Create library
add_library(${PROJECT_NAME} SHARED ${LIB_SOURCES} ${LIB_HEADERS})

# find packages
find_package(Eigen3 REQUIRED)    
find_package(TBB REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system)

# Common includes and links
set(COMMON_INCLUDES
    ${Eigen3_INCLUDE_DIRS} 
    ${Boost_INCLUDE_DIRS} 
    ${TBB_INCLUDE_DIRS} 
    ${Sophus_INCLUDE_DIRS})
    
set(COMMON_LIBRARIES 
    Eigen3::Eigen 
    ${Boost_LIBRARIES}  
    ${Sophus_LIBRARIES}
    Boost::thread
    Boost::system
    TBB::tbb
    pthread
    )

# Link common includes and libraries to the library
target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${PROJECT_NAME} PRIVATE ${COMMON_LIBRARIES} )

# executable setup
add_executable(${PROJECT_NAME}_exec src/run.cpp) 
target_link_libraries(${PROJECT_NAME}_exec PRIVATE ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME}_exec PRIVATE ${COMMON_INCLUDES})
target_link_libraries(${PROJECT_NAME}_exec PRIVATE ${COMMON_LIBRARIES})

# Debug and release configuration
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug mode")
    # target_compile_options(${PROJECT_NAME} PRIVATE -g -O0 -Wall -Wextra -Werror)
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0)
    # target_compile_options(${PROJECT_NAME}_exec PRIVATE -g -O0 -Wall -Wextra -Werror)
    target_compile_options(${PROJECT_NAME}_exec PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release mode")
    target_compile_options(${PROJECT_NAME} PRIVATE -O3 -march=native -DNDEBUG)
    target_compile_options(${PROJECT_NAME}_exec PRIVATE -O3 -march=native -DNDEBUG)
endif()


# Set the output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(${PROJECT_NAME}_exec PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
