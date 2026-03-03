cmake_minimum_required(VERSION 3.21)
project(gl3_wp19)

set(CMAKE_CXX_STANDARD 17)

# Set variables for source files
file(GLOB_RECURSE INCLUDE_FILES CONFIGURE_DEPENDS
        extern/tracy/public/tracy/Tracy.hpp
)

add_subdirectory(extern/glad)
add_subdirectory(extern/glfw)
add_subdirectory(extern/glm)
add_subdirectory(extern/soloud)
add_subdirectory(extern/tinygltf)
add_definitions(-DNOMINMAX)
add_subdirectory(extern/bullet3-3.25)

add_subdirectory(game)
