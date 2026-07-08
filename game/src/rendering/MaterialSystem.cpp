#include "MaterialSystem.h"
#include <iostream>
#include <vector>
#include <cstring>

// stb_image
#include "../../../extern/stb_image.h"
#include "../../../extern/stb_image_resize2.h"
#include <glad/glad.h>

namespace gl3 {

    static void fillSolidRGBA(std::vector<unsigned char>& out, int w, int h,
                              unsigned char r, unsigned char g, unsigned char b, unsigned char a)
    {
        out.resize(size_t(w) * size_t(h) * 4);
        for (int i = 0; i < w * h; ++i) {
            out[i*4+0] = r;
            out[i*4+1] = g;
            out[i*4+2] = b;
            out[i*4+3] = a;
        }
    }

    static bool isCacheUpToDate(const std::string& sourcePath, const std::string& cachedPath)
    {
        namespace fs = std::filesystem;
        if (!fs::exists(sourcePath) || !fs::exists(cachedPath)) return false;
        return fs::last_write_time(cachedPath) >= fs::last_write_time(sourcePath);
    }

    static GLuint createArrayTexRGBA8(int w, int h, int layers)
    {
        GLuint tex = 0;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex);

        glTexImage3D(GL_TEXTURE_2D_ARRAY,
                     0,
                     GL_RGBA8,
                     w, h, layers,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     nullptr);

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        return tex;
    }

    static void uploadLayerRGBA8(GLuint tex, int w, int h, int layer, const unsigned char* data)
    {
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                        0,
                        0, 0, layer,
                        w, h, 1,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        data);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    }

    // Shared loader for one map type with per-map fallback color
    static bool buildArrayFromPaths(
            GLuint tex,
            int width, int height, int layerCount,
            const std::array<std::string, MaterialSystem::kMaxMaterials>& paths,
            unsigned char fr, unsigned char fg, unsigned char fb, unsigned char fa,
            int forceChannelsRGBA,
            MaterialSystem* selfForHelpers)
    {
        stbi_set_flip_vertically_on_load(true);

        for (int layer = 0; layer < layerCount; ++layer) {
            const std::string& sourcePath = paths[layer];

            // Empty path => fallback
            if (sourcePath.empty()) {
                std::vector<unsigned char> fallback;
                fillSolidRGBA(fallback, width, height, fr, fg, fb, fa);
                uploadLayerRGBA8(tex, width, height, layer, fallback.data());
                continue;
            }

            const std::string cachedPath = selfForHelpers->buildResizedCachePath(sourcePath, width, height);

            int lw = 0, lh = 0, ln = 0;
            unsigned char* img = nullptr;
            bool loadedFromCache = false;

            // 1) Try cache
            if (selfForHelpers->fileExists(cachedPath) && isCacheUpToDate(sourcePath, cachedPath)) {
                img = stbi_load(cachedPath.c_str(), &lw, &lh, &ln, forceChannelsRGBA);
                if (img && lw == width && lh == height) {
                    loadedFromCache = true;
                } else {
                    if (img) {
                        stbi_image_free(img);
                        img = nullptr;
                    }
                }
            }

            // 2) Load source
            if (!img) {
                img = stbi_load(sourcePath.c_str(), &lw, &lh, &ln, forceChannelsRGBA);
                if (!img) {
                    std::cout << "MaterialSystem: failed to load: " << sourcePath
                              << " (fallback layer used)\n";
                    std::vector<unsigned char> fallback;
                    fillSolidRGBA(fallback, width, height, fr, fg, fb, fa);
                    uploadLayerRGBA8(tex, width, height, layer, fallback.data());
                    continue;
                }
            }

            // 3) Upload (resize if needed)
            if (lw == width && lh == height) {
                uploadLayerRGBA8(tex, width, height, layer, img);

                if (!loadedFromCache) {
                    selfForHelpers->saveRGBA8PNG(cachedPath, width, height, img);
                }

                stbi_image_free(img);
            } else {
                auto resized = selfForHelpers->resizeToRGBA(img, lw, lh, width, height);
                selfForHelpers->saveRGBA8PNG(cachedPath, width, height, resized.data());
                uploadLayerRGBA8(tex, width, height, layer, resized.data());
                stbi_image_free(img);
            }
        }

        glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
        return true;
    }

    bool MaterialSystem::initAllTextureArraysFromFiles(
            const std::array<std::string, kMaxMaterials>& albedoPaths,
            const std::array<std::string, kMaxMaterials>& normalPaths,
            const std::array<std::string, kMaxMaterials>& roughPaths,
            const std::array<std::string, kMaxMaterials>& aoPaths,
            const std::array<std::string, kMaxMaterials>& heightPaths,
            int forceChannelsRGBA)
    {
        destroy();

        layerCount = kMaxMaterials;
        width = 1024;
        height = 1024;

        albedoArrayTex = createArrayTexRGBA8(width, height, layerCount);
        normalArrayTex = createArrayTexRGBA8(width, height, layerCount);
        roughArrayTex  = createArrayTexRGBA8(width, height, layerCount);
        aoArrayTex     = createArrayTexRGBA8(width, height, layerCount);
        heightArrayTex = createArrayTexRGBA8(width, height, layerCount);

        // Albedo fallback: magenta (debug-visible)
        buildArrayFromPaths(albedoArrayTex, width, height, layerCount, albedoPaths,
                            255, 0, 255, 255, forceChannelsRGBA, this);

        // Normal fallback: flat normal
        buildArrayFromPaths(normalArrayTex, width, height, layerCount, normalPaths,
                            128, 128, 255, 255, forceChannelsRGBA, this);

        // Rough fallback: 1.0 (rough)
        buildArrayFromPaths(roughArrayTex, width, height, layerCount, roughPaths,
                            255, 255, 255, 255, forceChannelsRGBA, this);

        // AO fallback: 1.0 (no ambient darkening)
        buildArrayFromPaths(aoArrayTex, width, height, layerCount, aoPaths,
                            255, 255, 255, 255, forceChannelsRGBA, this);

        // Height fallback: 0.5 (neutral midpoint)
        buildArrayFromPaths(heightArrayTex, width, height, layerCount, heightPaths,
                            128, 128, 128, 255, forceChannelsRGBA, this);

        std::cout << "MaterialSystem: built ALL texture arrays "
                  << width << "x" << height
                  << " layers=" << layerCount << "\n";
        return true;
    }


    void MaterialSystem::destroy()
    {
        if (albedoArrayTex != 0) {
            glDeleteTextures(1, &albedoArrayTex);
            albedoArrayTex = 0;
        }
        if (normalArrayTex != 0) {
            glDeleteTextures(1, &normalArrayTex);
            normalArrayTex = 0;
        }
        if (roughArrayTex != 0) {
            glDeleteTextures(1, &roughArrayTex);
            roughArrayTex = 0;
        }
        if (aoArrayTex != 0) {
            glDeleteTextures(1, &aoArrayTex);
            aoArrayTex = 0;
        }
        if (heightArrayTex != 0) {
            glDeleteTextures(1, &heightArrayTex);
            heightArrayTex = 0;
        }
        layerCount = 0;
        width = height = 0;
    }

    std::vector<unsigned char> MaterialSystem::resizeToRGBA(
            const unsigned char* src, int sw, int sh,
            int dw, int dh)
    {
        std::vector<unsigned char> out(size_t(dw) * size_t(dh) * 4);
        stbir_resize_uint8_linear(
                src, sw, sh, 0,
                out.data(), dw, dh, 0,
                STBIR_RGBA
        );
        return out;
    }

    std::string MaterialSystem::buildResizedCachePath(const std::string& originalPath, int w, int h)
    {
        namespace fs = std::filesystem;
        fs::path p(originalPath);

        fs::path dir = p.parent_path();
        std::string stem = p.stem().string();

        fs::path cachedDir = dir / "cache";
        std::filesystem::create_directories(cachedDir);
        fs::path cached = cachedDir / (stem + "_resized_" + std::to_string(w) + "x" + std::to_string(h) + ".png");
        return cached.string();
    }

    bool MaterialSystem::fileExists(const std::string& path)
    {
        return std::filesystem::exists(path);
    }

    bool MaterialSystem::saveRGBA8PNG(const std::string& path, int w, int h, const unsigned char* data)
    {
        const int strideBytes = w * 4;
        int ok = stbi_write_png(path.c_str(), w, h, 4, data, strideBytes);
        return ok != 0;
    }

} // namespace gl3