cmake_minimum_required(VERSION 3.21)
project(VoxelEngine)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(extern/glad)
add_subdirectory(extern/glfw)
add_subdirectory(extern/glm)
add_subdirectory(extern/soloud)
add_subdirectory(extern/tinygltf)

add_definitions(-DNOMINMAX)

add_subdirectory(game)
