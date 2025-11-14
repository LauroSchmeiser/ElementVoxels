cmake_minimum_required(VERSION 3.18)

# set executable name
set(EXE_FILE Game)

# set variables for source files
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        src/rendering/*.cpp
        src/rendering/*.h
        src/entities/*.cpp
        src/entities/*.h
)
add_executable(${EXE_FILE} ${SOURCE_FILES}
        src/rendering/VoxelRenderer.cpp
        src/rendering/VoxelRenderer.h
        src/rendering/VoxelMesher.cpp
        src/rendering/VoxelMesher.h
        ../main.cpp
        src/rendering/TestChunk.cpp
        src/rendering/TestChunk.h
        src/entities/VoxelEntity.cpp
        src/entities/VoxelEntity.h
        src/rendering/smoothMesher.h
        src/rendering/smoothMesher.cpp)



# require C++ 17 compiler
target_compile_features(${EXE_FILE} PRIVATE cxx_std_17)

# link with dependencies
target_link_libraries(${EXE_FILE} PRIVATE glad glfw glm soloud)

target_compile_definitions(${EXE_FILE} PRIVATE ASSET_ROOT=assets)

# copy assets to output directory before build
add_custom_command(TARGET ${EXE_FILE} PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${EXE_FILE}>/assets)
