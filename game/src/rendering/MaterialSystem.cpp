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

        // Load first to get size
        int w = 0, h = 0, n = 0;
        unsigned char* first = stbi_load(materialPaths[0].c_str(), &w, &h, &n, forceChannelsRGBA);
        if (!first) {
            std::cout << "MaterialSystem: failed to load: " << materialPaths[0] << "\n";
            return false;
        }

        width = w;
        height = h;

        glGenTextures(1, &albedoArrayTex);
        glBindTexture(GL_TEXTURE_2D_ARRAY, albedoArrayTex);

        int tw=0, th=0, td=0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH,  &tw);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT, &th);
        glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH,  &td);

        std::cout << "MaterialSystem: GL_TEXTURE_2D_ARRAY allocated w=" << tw
                  << " h=" << th << " layers=" << td << "\n";

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cout << "MaterialSystem: GL error after creating array: " << err << "\n";
        }

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

        // Upload layer 0
        glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                        0,
                        0, 0, 0,
                        width, height, 1,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        first);

        stbi_image_free(first);

        // Upload remaining layers
        for (int layer = 1; layer < layerCount; ++layer) {
            int lw = 0, lh = 0, ln = 0;
            unsigned char* img = stbi_load(materialPaths[layer].c_str(), &lw, &lh, &ln, forceChannelsRGBA);
            if (!img) {
                std::cout << "MaterialSystem: failed to load: " << materialPaths[layer]
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

            if (lw != width || lh != height) {
                std::cout << "MaterialSystem: size mismatch for " << materialPaths[layer]
                          << " expected " << width << "x" << height
                          << " got " << lw << "x" << lh
                          << " (resizing texture)\n";

                auto resized = resizeToRGBA(img, lw, lh, width, height);

                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0,
                                0, 0, layer,
                                width, height, 1,
                                GL_RGBA, GL_UNSIGNED_BYTE,
                                resized.data());

                stbi_image_free(img);
                continue;
            }

            glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                            0,
                            0, 0, layer,
                            width, height, 1,
                            GL_RGBA,
                            GL_UNSIGNED_BYTE,
                            img);

            stbi_image_free(img);
        }

        // Sampler state
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        // Mips
        glGenerateMipmap(GL_TEXTURE_2D_ARRAY);

        /*
        // Optional anisotropic filtering
        if (GLAD_GL_EXT_texture_filter_anisotropic) {
            GLfloat maxAniso = 1.0f;
            glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &maxAniso);
            glTexParameterf(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAX_ANISOTROPY_EXT, maxAniso);
        }*/

        glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

        std::cout << "MaterialSystem: built texture array " << width << "x" << height
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
} // namespace gl3