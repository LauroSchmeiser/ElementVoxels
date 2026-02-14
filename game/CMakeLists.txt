cmake_minimum_required(VERSION 3.18)
project(VoxelGame)

# Executable name
set(EXE_FILE Game)

# Set variables for source files
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
        src/rendering/*.cpp
        src/rendering/*.h
        src/entities/*.cpp
        src/entities/*.h
)

# Create executable
add_executable(${EXE_FILE} ${SOURCE_FILES}
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
        src/physics/SpellPhysicsManager.cpp
        src/physics/SpellPhysicsManager.h
        src/physics/VoxelPhysicsBody.h
        src/physics/VoxelPhysicsManager.h
        src/physics/VoxelPhysicsManager.cpp
)

# Require C++ 17 compiler
target_compile_features(${EXE_FILE} PRIVATE cxx_std_17)

# Link with 3rd party dependencies (non-Bullet)
target_link_libraries(${EXE_FILE} PRIVATE
        glad
        glfw
        glm
        soloud
        Bullet3Collision
        Bullet3Common
        Bullet3Dynamics
        LinearMath
)

include(../extern/bullet3-3.25/UseBullet.cmake)

# --- Bullet detection + linking (robust) ---
# Try to find an installed Bullet first (config mode)
find_package(Bullet 3.25 QUIET)

if (Bullet_FOUND AND TARGET Bullet::BulletDynamics)
    message(STATUS "Using installed Bullet targets from find_package")
    target_link_libraries(${EXE_FILE} PRIVATE
            Bullet::BulletDynamics
            Bullet::BulletCollision
            Bullet3Common
            Bullet::LinearMath

    )
else()
    message(STATUS "Using in-tree or manually-found Bullet libs (fallback)")

    # If you added add_subdirectory(extern/bullet3-3.25) in the top-level CMake,
    # the in-tree targets should be available as these names:
    if (TARGET Bullet3Dynamics AND TARGET Bullet3Collision AND TARGET LinearMath)
        target_link_libraries(${EXE_FILE} PRIVATE
                Bullet3Dynamics
                Bullet3Collision
                LinearMath
                Bullet3Common  # optional if present and needed
        )
        # headers are propagated by target include dirs if Bullet target sets them;
        # otherwise set the include dir to source/install tree:
        target_include_directories(${EXE_FILE} PRIVATE
                "${CMAKE_SOURCE_DIR}/extern/bullet3-3.25/src"
                "${CMAKE_SOURCE_DIR}/extern/bullet3-3.25/install/include"
        )
    else()
        # As a last fallback, try to locate the library files under common paths
        set(_cand "${CMAKE_SOURCE_DIR}/extern/bullet3-3.25")
        set(_libdirs
                "${_cand}/install/lib"
                "${_cand}/install/lib/Debug"
                "${_cand}/build/lib"
                "${_cand}/build/lib/Debug"
        )
        find_library(_lmath NAMES LinearMath LinearMath_Debug PATHS ${_libdirs} NO_DEFAULT_PATH)
        find_library(_bcoll NAMES BulletCollision BulletCollision_Debug PATHS ${_libdirs} NO_DEFAULT_PATH)
        find_library(_bdyn  NAMES BulletDynamics BulletDynamics_Debug PATHS ${_libdirs} NO_DEFAULT_PATH)
        find_library(_b3c   NAMES Bullet3Common Bullet3Common_Debug PATHS ${_libdirs} NO_DEFAULT_PATH)

        if (_lmath AND _bdyn AND _bcoll)
            message(STATUS "Found Bullet libs: ${_bdyn} ${_bcoll} ${_lmath}")
            target_link_libraries(${EXE_FILE} PRIVATE "${_b3c}" "${_bcoll}" "${_bdyn}" "${_lmath}")
            if (EXISTS "${_cand}/install/include")
                target_include_directories(${EXE_FILE} PRIVATE "${_cand}/install/include")
            elseif (EXISTS "${_cand}/src")
                target_include_directories(${EXE_FILE} PRIVATE "${_cand}/src")
            endif()
        else()
            message(WARNING "Bullet not found. Physics features will be disabled.")
        endif()
    endif()
endif()
# --- end Bullet detection ---

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