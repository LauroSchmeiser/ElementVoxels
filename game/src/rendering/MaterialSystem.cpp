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

    bool MaterialSystem::initTextureArrayFromFiles(const std::vector<std::string>& materialPaths,
                                                   int forceChannelsRGBA)
    {
        destroy();

        stbi_set_flip_vertically_on_load(true);

        layerCount = (int)materialPaths.size();
        if (layerCount <= 0) {
            std::cout << "MaterialSystem: no material textures provided.\n";
            return false;
        }
        if (layerCount > kMaxMaterials) {
            std::cout << "MaterialSystem: too many materials (" << layerCount
                      << "), max is " << kMaxMaterials << "\n";
            return false;
        }

        // Force canonical texture size for the array
        width = 1024;
        height = 1024;

        glGenTextures(1, &albedoArrayTex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, albedoArrayTex);

        // Allocate storage for all layers (RGBA8)
        glTexImage3D(GL_TEXTURE_2D_ARRAY,
                     0,
                     GL_RGBA8,
                     width,
                     height,
                     layerCount,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     nullptr);

        for (int layer = 0; layer < layerCount; ++layer) {
            const std::string& sourcePath = materialPaths[layer];
            const std::string cachedPath = buildResizedCachePath(sourcePath, width, height);

            int lw = 0, lh = 0, ln = 0;
            unsigned char* img = nullptr;
            bool loadedFromCache = false;

            // 1) Try cached resized PNG first
            if (fileExists(cachedPath) && isCacheUpToDate(sourcePath, cachedPath)) {
                img = stbi_load(cachedPath.c_str(), &lw, &lh, &ln, forceChannelsRGBA);
                if (img && lw == width && lh == height) {
                    loadedFromCache = true;
                    std::cout << "MaterialSystem: loaded cached texture: " << cachedPath << "\n";
                } else {
                    if (img) {
                        stbi_image_free(img);
                        img = nullptr;
                    }
                }
            }

            // 2) If no valid cache, load original
            if (!img) {
                img = stbi_load(sourcePath.c_str(), &lw, &lh, &ln, forceChannelsRGBA);
                if (!img) {
                    std::cout << "MaterialSystem: failed to load: " << sourcePath
                              << " (filling fallback magenta)\n";

                    std::vector<unsigned char> fallback;
                    fillSolidRGBA(fallback, width, height, 255, 0, 255, 255);

                    glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                                    0,
                                    0, 0, layer,
                                    width, height, 1,
                                    GL_RGBA,
                                    GL_UNSIGNED_BYTE,
                                    fallback.data());
                    continue;
                }
            }

            // 3) If image is already correct size, upload directly
            if (lw == width && lh == height) {
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                                0,
                                0, 0, layer,
                                width, height, 1,
                                GL_RGBA,
                                GL_UNSIGNED_BYTE,
                                img);

                // If it came from original and not cache, optionally create cache once
                if (!loadedFromCache) {
                    if (saveRGBA8PNG(cachedPath, width, height, img)) {
                        std::cout << "MaterialSystem: cached original-sized texture as PNG: "
                                  << cachedPath << "\n";
                    } else {
                        std::cout << "MaterialSystem: failed to write cached PNG: "
                                  << cachedPath << "\n";
                    }
                }

                stbi_image_free(img);
                continue;
            }

            // 4) Wrong size: resize, save cache, upload resized
            std::cout << "MaterialSystem: resizing " << sourcePath
                      << " from " << lw << "x" << lh
                      << " to " << width << "x" << height << "\n";

            auto resized = resizeToRGBA(img, lw, lh, width, height);

            if (saveRGBA8PNG(cachedPath, width, height, resized.data())) {
                std::cout << "MaterialSystem: wrote cached resized texture: "
                          << cachedPath << "\n";
            } else {
                std::cout << "MaterialSystem: failed to write cached resized texture: "
                          << cachedPath << "\n";
            }

            glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                            0,
                            0, 0, layer,
                            width, height, 1,
                            GL_RGBA,
                            GL_UNSIGNED_BYTE,
                            resized.data());

            stbi_image_free(img);
        }

        // Sampler state
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        std::cout << "MaterialSystem: built texture array "
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