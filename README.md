# ElementVoxels (VoxelEngine / VoxelGame)

**ElementVoxels** is my personal project, where I’m building a **custom voxel renderer + game engine** toward a game concept focused on “using the world around you”.  
This repo contains the engine layer (**VoxelEngine**) and the game layer (**VoxelGame / Game**), with rendering, physics-style voxel interactions, scenes/UI, and profiling instrumentation.

> Status: actively developed / experimental. APIs and content formats are expected to change.

---

## Highlights

- **C++17** engine + game architecture (CMake)
- **OpenGL renderer** (GLAD + GLFW + GLSL shaders)
- **Voxel world systems**
  - Chunk/ChunkManager + multigrid style management
  - GPU data layouts (std430 structs)
  - Marching tables / voxel structures groundwork
- **Gameplay/Systems**
  - Character controller
  - Voxel physics manager + voxel bodies
  - Carving / crater stamping (fast carving utilities)
  - Async spell casting experiments
- **UI**
  - ImGui layer + docking (via vcpkg)
  - Multiple scenes: Main Menu, Loading, Gameplay, Game Over
- **Profiling**
  - Tracy client enabled (+ GPU profiling)

---

## Tech Stack / Dependencies

### Build system
- **CMake** (root requires 3.21, game module requires 3.18)
- **vcpkg** for select libraries (Tracy, ImGui, glfw3)

### Libraries
- `glad` (OpenGL loader) — vendored via `extern/`
- `glfw` — vendored via `extern/` (+ also referenced in vcpkg dependencies; see notes below)
- `glm` — vendored via `extern/`
- `soloud` — vendored via `extern/`
- `tinygltf` — vendored via `extern/`
- `imgui` — via vcpkg (`glfw-binding`, `opengl3-binding`, `docking-experimental`)
- `tracy` — via vcpkg (with crash-handler, pinned override 0.11.1)

---

## Repository Layout (high level)

- `game/`  
  Game executable target (**Game**) + gameplay code and assets integration
- `src/`  
  Core code (rendering, entities, input, physics, spells, scenes, UI)
- `assets/`  
  Runtime assets copied to the build output directory by CMake

---

## Building (Windows / macOS / Linux)

This project is intended to be built with **CMake + vcpkg**.

### 0) Prerequisites
- A C++17 compiler
- CMake **3.21+** (root `CMakeLists.txt` uses 3.21)
- vcpkg installed and available

### 1) Install vcpkg dependencies
This repo includes a `vcpkg.json` manifest. Typical usage is **manifest mode**.

You’ll need (as defined in `vcpkg.json`):
- tracy (crash-handler)
- imgui (glfw-binding, opengl3-binding, docking-experimental)
- glfw3

### 2) Configure with CMake
Example (adjust triplet/toolchain paths for your system):

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE="<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake"
```

### 3) Build
```bash
cmake --build build --config Release
```

### 4) Run
The executable target is named:
- **Game**

At build time, assets are copied into the build output directory via a `copyAssets` custom target, and the code uses:

- `ASSET_ROOT=assets`

So the build output folder should contain an `assets/` directory alongside the executable.

---

## Notes on dependencies (vendored + vcpkg)

This repo uses `add_subdirectory(extern/...)` for several libraries (glad/glfw/glm/soloud/tinygltf), and also includes `glfw3` in `vcpkg.json`.

If you run into linking conflicts because GLFW is being provided twice, you have two common options:
- Prefer the **vendored** version and remove/ignore the vcpkg GLFW dependency, or
- Prefer **vcpkg** and remove/disable the vendored GLFW subdirectory for that configuration

---

## Media

[WORK IN PROGRESS]

---

## Roadmap

- [ ] Finish base-gameplay Loop
- [ ] Add simple enemies for PvE Mode
- [ ] Package a playable demo build via GitHub Releases
- [ ] Add voxel-type based-interactions for physics
- [ ] Add voxel-type based-rendering for fluids and gases
- [ ] Add material interactions
- [ ] Optimizations for chunk based-system
- [ ] Look into Multiplayer
- [ ] Package a playable demo build via GitHub Releases

---

## Credits / Third-Party

This project uses third-party libraries (see Tech Stack).  

---

## License
All Rights Reserved
