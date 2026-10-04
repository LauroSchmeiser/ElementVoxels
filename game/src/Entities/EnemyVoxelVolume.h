#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "../rendering/VoxelStructures.h"

namespace gl3 {

    struct LocalVoxelVolume {
        glm::ivec3 dims = {33,33,33};
        float voxelSize = VOXEL_SIZE;

        struct Corner {
            float density = -1000.0f;
            glm::vec3 color = glm::vec3(1);
            uint32_t material = 0;
            uint8_t type = 0;
        };

        std::vector<Corner> corners;

        LocalVoxelVolume() {
            corners.assign((size_t)dims.x * dims.y * dims.z, {});
        }

        void init(glm::ivec3 cornerDims, float vs) {
            dims = cornerDims;
            voxelSize = vs;
            corners.assign((size_t)dims.x * dims.y * dims.z, {});
        }
        inline size_t idx(int x,int y,int z) const {
            return (size_t)x + (size_t)y * dims.x + (size_t)z * dims.x * dims.y;
        }

        Corner& at(int x,int y,int z) {
            assert(x >= 0 && x < dims.x);
            assert(y >= 0 && y < dims.y);
            assert(z >= 0 && z < dims.z);
            assert(corners.size() == (size_t)dims.x * dims.y * dims.z);
            return corners[idx(x,y,z)];
        }

        [[nodiscard]] const Corner& at(int x,int y,int z) const {
            assert(x >= 0 && x < dims.x);
            assert(y >= 0 && y < dims.y);
            assert(z >= 0 && z < dims.z);
            assert(corners.size() == (size_t)dims.x * dims.y * dims.z);
            return corners[idx(x,y,z)];
        }

        // Simple “fill a sphere” in local space (for initial enemy body)
        void fillSphere(glm::vec3 centerLocal, float radiusWorld, glm::vec3 col, uint32_t material=0, uint8_t type=1) {
            const float r2 = radiusWorld * radiusWorld;
            for (int z=0; z<dims.z; ++z)
                for (int y=0; y<dims.y; ++y)
                    for (int x=0; x<dims.x; ++x) {
                        glm::vec3 p = glm::vec3(x,y,z) * voxelSize;
                        float d2 = glm::dot(p-centerLocal, p-centerLocal);
                        Corner& c = at(x,y,z);

                        // “density” convention: positive inside
                        float s = radiusWorld - std::sqrt(std::max(0.0f, d2));
                        c.density = s;
                        if (s >= -1.0f) {
                            c.color = col;
                            c.material = material;
                            c.type = type;
                        }
                    }
        }

        // Damage carve: subtract density inside sphere => “removes” matter
        void carveSphere(glm::vec3 centerLocal, float radiusWorld, float strength) {
            if (!isInitialized()) {
                init(dims,VOXEL_SIZE);
                //assert(false && "LocalVoxelVolume used before init");
                return;
            }

            const float r2 = radiusWorld * radiusWorld;
            for (int z=0; z<dims.z; ++z)
                for (int y=0; y<dims.y; ++y)
                    for (int x=0; x<dims.x; ++x) {
                        glm::vec3 p = glm::vec3(x,y,z) * voxelSize;
                        float d2 = glm::dot(p-centerLocal, p-centerLocal);
                        if (d2 > r2) continue;
                        at(x,y,z).density -= strength;
                    }
        }

        static std::vector<LocalVoxelVolume> splinterSphere(
                const LocalVoxelVolume& body,
                glm::vec3 centerLocal,
                float density,
                float strength,
                glm::vec3 localForce,
                uint32_t maxBodies,
                std::vector<glm::vec3>* outLocalImpulses = nullptr)
        {
            assert(body.isInitialized());

            if (outLocalImpulses) {
                outLocalImpulses->clear();
            }

            if (!body.isInitialized() || maxBodies == 0) {
                return {};
            }

            // Derive the desired count, with a mandatory safety cap.
            const float safeDensity = glm::max(density, 0.001f);
            const uint32_t splinterCount = glm::clamp(
                    static_cast<uint32_t>(glm::ceil(strength / safeDensity)),
                    1u,
                    maxBodies);

            // No split required.
            if (splinterCount == 1) {
                if (outLocalImpulses) {
                    outLocalImpulses->push_back(localForce);
                }
                return { body };
            }

            // The fractures fan out around the incoming force direction.
            glm::vec3 fractureAxis = localForce;
            if (glm::dot(fractureAxis, fractureAxis) < 0.000001f) {
                fractureAxis = glm::vec3(0.0f, 1.0f, 0.0f);
            } else {
                fractureAxis = glm::normalize(fractureAxis);
            }

            // Create two axes perpendicular to fractureAxis. Fracture wedges are
            // distributed around these axes.
            const glm::vec3 helper =
                    glm::abs(fractureAxis.y) < 0.95f
                    ? glm::vec3(0.0f, 1.0f, 0.0f)
                    : glm::vec3(1.0f, 0.0f, 0.0f);

            const glm::vec3 tangentU = glm::normalize(glm::cross(helper, fractureAxis));
            const glm::vec3 tangentV = glm::normalize(glm::cross(fractureAxis, tangentU));

            constexpr float TWO_PI = 6.28318530718f;
            const float sectorAngle = TWO_PI / static_cast<float>(splinterCount);

            std::vector<LocalVoxelVolume> result;
            result.reserve(splinterCount);

            if (outLocalImpulses) {
                outLocalImpulses->reserve(splinterCount);
            }

            for (uint32_t pieceIndex = 0; pieceIndex < splinterCount; ++pieceIndex) {
                const float startAngle = static_cast<float>(pieceIndex) * sectorAngle;
                const float endAngle = startAngle + sectorAngle;

                // The two rays define this piece's wedge in the plane perpendicular
                // to the force direction.
                const glm::vec2 startRay(glm::cos(startAngle), glm::sin(startAngle));
                const glm::vec2 endRay(glm::cos(endAngle), glm::sin(endAngle));

                LocalVoxelVolume piece;
                piece.init(body.dims, body.voxelSize);

                for (int z = 0; z < body.dims.z; ++z) {
                    for (int y = 0; y < body.dims.y; ++y) {
                        for (int x = 0; x < body.dims.x; ++x) {
                            const glm::vec3 position =
                                    glm::vec3(
                                            static_cast<float>(x),
                                            static_cast<float>(y),
                                            static_cast<float>(z)) * body.voxelSize;

                            const glm::vec3 relativePosition = position - centerLocal;

                            // Project this voxel onto the plane around the fracture axis.
                            const glm::vec2 projected(
                                    glm::dot(relativePosition, tangentU),
                                    glm::dot(relativePosition, tangentV));

                            // Signed distance-like values for the two cut planes.
                            // A point belongs to this counter-clockwise wedge when both
                            // values are >= 0.
                            const float afterStart =
                                    startRay.x * projected.y - startRay.y * projected.x;

                            const float beforeEnd =
                                    projected.x * endRay.y - projected.y * endRay.x;

                            const float wedgeDensity = glm::min(afterStart, beforeEnd);

                            const Corner& source = body.at(x, y, z);
                            Corner& destination = piece.at(x, y, z);

                            // Keep source appearance/material data.
                            destination = source;

                            // Intersect the original signed-density field with this
                            // fracture wedge. This preserves the original geometry,
                            // while creating a closed planar surface along each split.
                            destination.density = glm::min(source.density, wedgeDensity);

                            if (destination.density < -1.0f) {
                                destination.type = 0;
                            }
                        }
                    }
                }

                result.push_back(piece);

                if (outLocalImpulses) {
                    // Push every piece outward from the impact axis, while retaining
                    // an equal share of the incoming/explosion force.
                    const float middleAngle = startAngle + sectorAngle * 0.5f;

                    const glm::vec3 outwardDirection =
                            tangentU * glm::cos(middleAngle) +
                            tangentV * glm::sin(middleAngle);

                    const glm::vec3 impulse =
                            (localForce / static_cast<float>(splinterCount)) +
                            outwardDirection * (strength / static_cast<float>(splinterCount));

                    outLocalImpulses->push_back(impulse);
                }
            }

            return result;
        }

        void unionSphere(glm::vec3 centerLocal, float radiusWorld, glm::vec3 col, uint32_t material=0, uint8_t type=1) {
            for (int z=0; z<dims.z; ++z)
                for (int y=0; y<dims.y; ++y)
                    for (int x=0; x<dims.x; ++x) {
                        glm::vec3 p = glm::vec3(x,y,z) * voxelSize;
                        float s = radiusWorld - glm::length(p - centerLocal);

                        Corner& c = at(x,y,z);
                        if (s > c.density) {
                            c.density = s;
                            if (s >= -1.0f) {
                                c.color = col;
                                c.material = material;
                                c.type = type;
                            }
                        }
                    }
        }
        void setAllSolidToMaterial(uint32_t newMaterial) {
            for (auto& c : corners) {
                if (c.type > 0 && c.density > -1.0f) {
                    c.material = newMaterial;
                }
            }
        }

        bool isInitialized() const {
            return corners.size() == (size_t)dims.x * dims.y * dims.z && !corners.empty();
        }
    };
}