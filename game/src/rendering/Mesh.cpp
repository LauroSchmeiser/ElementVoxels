#include "Mesh.h"
#include "glad/glad.h"

namespace gl3 {
    template<typename T>
    GLuint createBuffer(GLuint bufferType, const std::vector<T> &bufferData) {
        unsigned int buffer = 0;
        glGenBuffers(1, &buffer);
        glBindBuffer(bufferType, buffer);
        glBufferData(bufferType, bufferData.size() * sizeof(T), bufferData.data(), GL_STATIC_DRAW);
        return buffer;
    }

    Mesh::Mesh(const std::vector<float> &vertices, const std::vector<unsigned int> &indices)
            : numberOfIndices(indices.size())
    {
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        VBO = createBuffer(GL_ARRAY_BUFFER, vertices);
        EBO = createBuffer(GL_ELEMENT_ARRAY_BUFFER, indices);

        // Vertex layout: position(3), color(3), normal(3)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)0);        // position
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(3 * sizeof(float))); // color
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * sizeof(float), (void*)(6 * sizeof(float))); // normal
        glEnableVertexAttribArray(2);

        glBindVertexArray(0); // unbind VAO
    }

    void Mesh::draw() const {
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, numberOfIndices, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }


    Mesh::~Mesh() {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteVertexArrays(1, &VAO);
    }
}