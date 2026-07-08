#pragma once
#include <glad/glad.h>
#include <string>
#include <vector>
#include <array>
#include <filesystem>
#include "../../stb_image_write.h"

namespace gl3 {

    struct MaterialParams {
        float roughness = 0.9f;
        float specular  = 0.05f;
        float uvScale   = 1.0f;
    };

    struct MaterialSystem {
        static constexpr int kMaxMaterials = 64;

        GLuint albedoArrayTex = 0;
        GLuint normalArrayTex = 0;
        GLuint roughArrayTex  = 0;
        GLuint aoArrayTex     = 0;
        GLuint heightArrayTex = 0;

        int layerCount = kMaxMaterials;
        int width = 0;
        int height = 0;

        std::array<MaterialParams, kMaxMaterials> params{};

        // New: all maps, fixed by material id [0..63], empty path => fallback layer
        bool initAllTextureArraysFromFiles(
                const std::array<std::string, kMaxMaterials>& albedoPaths,
                const std::array<std::string, kMaxMaterials>& normalPaths,
                const std::array<std::string, kMaxMaterials>& roughPaths,
                const std::array<std::string, kMaxMaterials>& aoPaths,
                const std::array<std::string, kMaxMaterials>& heightPaths,
                int forceChannelsRGBA = 4);

        std::vector<unsigned char> resizeToRGBA(
                const unsigned char* src, int sw, int sh,
                int dw, int dh);

        std::string buildResizedCachePath(const std::string& originalPath, int w, int h);
        bool fileExists(const std::string& path);
        bool saveRGBA8PNG(const std::string& path, int w, int h, const unsigned char* data);

        void destroy();
    };

} // namespace gl3