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
        src/rendering/*.cpp
        src/rendering/*.h
        src/entities/*.cpp
        src/entities/*.h
)

# Create executable with explicit file lists
add_executable(${EXE_FILE}
        ${SOURCE_FILES}
        src/rendering/VoxelStructures.h
        ../main.cpp
        src/rendering/marchingTables.h
        src/rendering/SunBillboard.cpp
        src/rendering/SunBillboard.h
        ../extern/robin_hood.h
        src/rendering/MultiGridChunkManager.h
        src/rendering/ChunkManager.cpp
        src/rendering/ChunkManager.h
        src/rendering/Chunk.h
        src/Input/InputManager.h
        src/Input/InputActionMap.h
        src/CharacterController.cpp
        src/CharacterController.h
        src/physics/VoxelPhysicsBody.h
        src/physics/VoxelPhysicsManager.h
        src/physics/VoxelPhysicsManager.cpp
        src/rendering/GpuStructsStd430.h
        src/physics/VoxelCarver.h
        src/physics/FastCraterCarver.h
        src/physics/CraterStampBatch.h
        src/spells/spellCastAsync.h
        src/spells/spellCastAsync.cpp
        src/rendering/MaterialSystem.h
        src/rendering/MaterialSystem.cpp
        src/rendering/stb_image_impl.cpp
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