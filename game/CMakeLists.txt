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
)



# require C++ 17 compiler
target_compile_features(${EXE_FILE} PRIVATE cxx_std_17)

# link with dependencies
target_link_libraries(${EXE_FILE} PRIVATE glad glfw glm soloud)

target_compile_definitions(${EXE_FILE} PRIVATE ASSET_ROOT=assets)

file(GLOB_RECURSE ASSET_FILES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/assets/*)
set(COPIED_FILES)
# copy assets to output directory before build
foreach (FILE IN LISTS ASSET_FILES)
    file(RELATIVE_PATH REL ${CMAKE_SOURCE_DIR} ${FILE})
    set(OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${REL}")
    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${CMAKE_COMMAND} -E copy ${FILE} ${OUTPUT_FILE}
            DEPENDS ${FILE})
    list(APPEND COPIED_FILES ${OUTPUT_FILE})
endforeach ()
add_custom_target(copyAssets DEPENDS ${COPIED_FILES})
add_dependencies(${EXE_FILE} copyAssets)
