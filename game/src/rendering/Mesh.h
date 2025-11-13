#pragma once
#include <vector>

namespace gl3 {
    class Mesh {
    public:
        explicit Mesh(const std::vector<float> &vertices, const std::vector<unsigned int> &indices);
        void draw() const;
        ~Mesh();

        // Delete copy constructor
        Mesh(const Mesh &other) = default;

        // Explicit move constructor
        Mesh(Mesh &&other) noexcept {
            std::swap(this->VBO, other.VBO);
            std::swap(this->EBO, other.EBO);
            std::swap(this->numberOfIndices, other.numberOfIndices);
        }

    private:
        unsigned int VBO = 0;
        unsigned int EBO = 0;
        unsigned int VAO = 0;
        unsigned int numberOfIndices = 0;
    };
}
