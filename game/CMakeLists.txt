cmake_minimum_required(VERSION 3.18)
project(VoxelGame)

# Executable name
set(EXE_FILE Game)

# Find Tracy package from vcpkg
find_package(Tracy CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)

# Set variables for source files
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
)

# Create executable with explicit file lists
add_executable(${EXE_FILE}
        ${SOURCE_FILES}
        ../main.cpp
        ../extern/robin_hood.h
)

# Link Tracy and enable it
target_link_libraries(${EXE_FILE} PRIVATE Tracy::TracyClient)
target_compile_definitions(${EXE_FILE} PRIVATE TRACY_ENABLE TRACY_GPU_ENABLE)


# Require C++ 17 compiler
target_compile_features(${EXE_FILE} PRIVATE cxx_std_17)

# Link with other 3rd party dependencies
target_link_libraries(${EXE_FILE} PRIVATE
        glad
        glfw
        glm
        soloud
        imgui::imgui
)

# Set asset root definition
target_compile_definitions(${EXE_FILE} PRIVATE ASSET_ROOT=assets)

# Copy assets to output directory
file(GLOB_RECURSE ASSET_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/assets/*)
set(COPIED_FILES)
foreach(FILE IN LISTS ASSET_FILES)
    file(RELATIVE_PATH REL ${CMAKE_SOURCE_DIR} ${FILE})
    set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL}")
    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy ${FILE} ${OUTPUT_FILE}
            DEPENDS ${FILE})
    list(APPEND COPIED_FILES ${OUTPUT_FILE})
endforeach()
add_custom_target(copyAssets DEPENDS ${COPIED_FILES})
add_dependencies(${EXE_FILE} copyAssets)