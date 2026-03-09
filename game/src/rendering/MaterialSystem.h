#pragma once
#include <glad/glad.h>
#include <string>
#include <vector>
#include <array>

namespace gl3 {

    struct MaterialParams {
        float roughness = 0.9f;
        float specular  = 0.05f;
        float uvScale   = 1.0f;
    };

    struct MaterialSystem {
        static constexpr int kMaxMaterials = 64;

        GLuint albedoArrayTex = 0;
        int layerCount = 0;
        int width = 0;
        int height = 0;

        std::array<MaterialParams, kMaxMaterials> params{};

        // Load 1 texture per material id: materialPaths[i] goes into layer i.
        // Layers beyond materialPaths.size() are filled with a fallback color.
        bool initTextureArrayFromFiles(const std::vector<std::string>& materialPaths,
                                       int forceChannelsRGBA = 4);

        void destroy();
    };

} // namespace gl3