cmake_minimum_required(VERSION 3.28)

project(kuwahara)

find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)
add_executable(debug src/main.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -O2 )
target_compile_options(debug PRIVATE -g)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
target_link_libraries(debug PRIVATE OpenCL::OpenCL)

set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

set_target_properties(debug	PROPERTIES C_STANDARD 11
                                       C_STANDARD_REQUIRED ON
                                       C_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_compile_definitions(debug PRIVATE CL_TARGET_OPENCL_VERSION=300)
