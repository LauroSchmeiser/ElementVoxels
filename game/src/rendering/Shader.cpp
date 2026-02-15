#include "Shader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include "glm/gtc/type_ptr.hpp"
#include "../Assets.h"


namespace gl3 {
    struct glStatusData {
        int success;
        const char *shaderName;
        char infoLog[GL_INFO_LOG_LENGTH];
    };

    Shader::Shader(const fs::path &vertexShaderPath, const fs::path &fragmentShaderPath)
            : shaderProgram(0), vertexShader(0), fragmentShader(0), computeShader(0){
        vertexShader = loadAndCompileShader(GL_VERTEX_SHADER, vertexShaderPath);
        fragmentShader = loadAndCompileShader(GL_FRAGMENT_SHADER, fragmentShaderPath);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        GLint success = 0;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

        if (!success) {
            GLint maxLength = 0;
            glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &maxLength);

            std::vector<char> errorLog(maxLength);
            glGetProgramInfoLog(shaderProgram, maxLength, &maxLength, &errorLog[0]);

            std::cout << "COMPUTE SHADER LINK ERROR:\n" << errorLog.data() << std::endl;
        }

        glDetachShader(shaderProgram, vertexShader);
        glDetachShader(shaderProgram, fragmentShader);
    }

    Shader::Shader(const fs::path &computeShaderPath)
            : shaderProgram(0), vertexShader(0), fragmentShader(0), computeShader(0){
        vertexShader = 0;
        fragmentShader = 0;
        computeShader = loadAndCompileShader(GL_COMPUTE_SHADER, computeShaderPath);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, computeShader);
        glLinkProgram(shaderProgram);
        GLint success = 0;
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

        if (!success) {
            GLint maxLength = 0;
            glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &maxLength);

            std::vector<char> errorLog(maxLength);
            glGetProgramInfoLog(shaderProgram, maxLength, &maxLength, &errorLog[0]);

            std::cout << "COMPUTE SHADER LINK ERROR:\n" << errorLog.data() << std::endl;
        }

        glDetachShader(shaderProgram, computeShader);
    }


    unsigned int Shader::loadAndCompileShader(GLuint shaderType, const fs::path &shaderPath) {
        auto shaderSource = readText(shaderPath);
        auto source = shaderSource.c_str();
        auto shaderID = glCreateShader(shaderType);
        glShaderSource(shaderID, 1, &source, nullptr);
        glCompileShader(shaderID);

        glStatusData compilationStatus{};

        if (shaderType == GL_VERTEX_SHADER) compilationStatus.shaderName = "Vertex";
        else if (shaderType == GL_FRAGMENT_SHADER) compilationStatus.shaderName = "Fragment";
        else if (shaderType == GL_COMPUTE_SHADER) compilationStatus.shaderName = "Compute";

        glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compilationStatus.success);
        if (compilationStatus.success == GL_FALSE) {
            GLint logLen = 0;
            glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &logLen);
            std::string log(logLen, '\0');
            glGetShaderInfoLog(shaderID, logLen, nullptr, &log[0]);

            std::cerr << "Shader compile failed (" << shaderPath << "):\n";
            std::cerr << "=== SOURCE ===\n" << shaderSource << "\n";
            std::cerr << "=== LOG ===\n" << log << "\n";
            throw std::runtime_error("Shader compilation failed: " + shaderPath.string());
        }

        return shaderID;
    }

    std::string Shader::readText(const fs::path &filePath) {
        std::ifstream sourceFile(resolveAssetPath(filePath));
        std::stringstream buffer;
        buffer << sourceFile.rdbuf();
        return buffer.str();
    }

    void Shader::setMatrix(const std::string &uniformName, glm::mat4 matrix) const {
        auto uniformLocation = glGetUniformLocation(shaderProgram, uniformName.c_str());
        glUniformMatrix4fv(uniformLocation, 1, GL_FALSE, glm::value_ptr(matrix));
    }

    void Shader::setVector(const std::string &uniformName, glm::vec4 vector) const {
        auto uniformLocation = glGetUniformLocation(shaderProgram, uniformName.c_str());
        glUniform4fv(uniformLocation, 1, glm::value_ptr(vector));
    }

    void Shader::setVec3(const std::string& uniformName, glm::vec3 vector) const {
        auto loc = glGetUniformLocation(shaderProgram, uniformName.c_str());
        glUniform3fv(loc, 1, glm::value_ptr(vector));
    }

    void Shader::setVec2(const std::string& uniformName, glm::vec2 vector) const {
        auto loc = glGetUniformLocation(shaderProgram, uniformName.c_str());
        glUniform2fv(loc, 1, glm::value_ptr(vector));
    }

    void Shader::setIVec3(const std::string& uniformName, glm::ivec3 vector) const {
        auto loc = glGetUniformLocation(shaderProgram, uniformName.c_str());
        if (loc == -1) {
            // optional: debug print
            std::cerr << "Warning: uniform '" << uniformName << "' not found (setIVec3)\n";
            return;
        }
        glUniform3i(loc, vector.x, vector.y, vector.z);
    }


    void Shader::setBool(const std::string &uniformName, bool value) const {
        auto uniformLocation = glGetUniformLocation(shaderProgram, uniformName.c_str());
        glUniform1i(uniformLocation, value ? 1 : 0);
    }
    void Shader::setFloat(const std::string &uniformName, float value) const {
        auto uniformLocation = glGetUniformLocation(shaderProgram, uniformName.c_str());
        if(uniformLocation==-1)
        {
            std::cout<< "This doesnt work: " << uniformName.c_str() << "\n" ;
        }
        glUniform1f(uniformLocation, value);
    }

    void Shader::setInt(const std::string &uniformName, int value) const
    {
        auto uniformLocation = glGetUniformLocation(shaderProgram, uniformName.c_str());
        if(uniformLocation==-1)
        {
            std::cout<< "This doesnt work: " << uniformName.c_str() << "\n" ;
        }
        glUniform1i(uniformLocation, value);
    }

    void Shader::setUInt(const std::string &uniformName, unsigned int value) const
    {
        GLint loc = glGetUniformLocation(shaderProgram, uniformName.c_str());
        if (loc == -1)
        {
            std::cout << "Warning: uniform '" << uniformName << "' not found in shader.\n";
            return;
        }

        glUniform1ui(loc, value);
    }

    void Shader::use() const {
        glUseProgram(shaderProgram);
    }

    Shader::~Shader() {
        // Delete all shader objects that were created
        if (vertexShader != 0) {
            glDeleteShader(vertexShader);
        }
        if (fragmentShader != 0) {
            glDeleteShader(fragmentShader);
        }
        if (computeShader != 0) {
            glDeleteShader(computeShader);
        }

        // Also delete the program
        if (shaderProgram != 0) {
            glDeleteProgram(shaderProgram);
        }
    }
}
