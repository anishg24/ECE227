cmake_minimum_required(VERSION 3.30)
project(influence_analysis CXX)
include(CMakePrintHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_custom_target(data data_files COMMAND make all WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../data)
file(GLOB DATA_FILES CONFIGURE_DEPENDS "data/*.txt")
file(COPY ${DATA_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data)

file(GLOB_RECURSE PROJECT_SOURCES "src/*.cpp")

find_package(Boost 1.70 COMPONENTS graph filesystem REQUIRED)

if (Boost_FOUND)
    message(STATUS "Boost found. Version: ${Boost_VERSION}")
    message(STATUS "Boost include directories: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost library directories: ${Boost_LIBRARY_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Boost not found! Please set BOOST_ROOT or BOOST_LIBRARYDIR.")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found, enabling support.")
else()
    message(STATUS "OpenMP not found, building without OpenMP support.")
endif()

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC "include")
target_link_libraries(${PROJECT_NAME} PRIVATE Boost::graph Boost::filesystem OpenMP::OpenMP_CXX)

message(STATUS ${CMAKE_INCLUDE_PATH})
cmake_print_properties(TARGETS ${PROJECT_NAME} PROPERTIES SOURCES INCLUDE_DIRECTORIES)