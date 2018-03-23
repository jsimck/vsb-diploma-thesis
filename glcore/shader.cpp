#include "shader.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace tless {
    void Shader::checkCompileErrors(GLuint shader, int type) {
        GLint success;
        GLchar infoLog[1024];

        // Check for errors
        if (type != TYPE_PROGRAM) {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(shader, 1024, nullptr, infoLog);
                std::cout << "Error - SHADER_COMPILATION_ERROR of type: " << (type == TYPE_VERTEX ? "TYPE_VERTEX" : "TYPE_FRAGMENT") << std::endl << infoLog << std::endl;
            }
        } else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(shader, 1024, nullptr, infoLog);
                std::cout << "Error - PROGRAM_LINKING_ERROR of type: TYPE_PROGRAM" << std::endl << infoLog << std::endl;
            }
        }
    }

    void Shader::init(const char *vertexShader, const char *fragmentShader) {
        // Compile shaders
        GLuint vertex, fragment;

        // Vertex Shader
        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &vertexShader, nullptr);
        glCompileShader(vertex);
        checkCompileErrors(vertex, TYPE_VERTEX);

        // Fragment Shader
        fragment = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment, 1, &fragmentShader, nullptr);
        glCompileShader(fragment);
        checkCompileErrors(fragment, TYPE_FRAGMENT);

        // Create Shader Program
        id = glCreateProgram();
        glAttachShader(id, vertex);
        glAttachShader(id, fragment);
        glLinkProgram(id);
        checkCompileErrors(id, TYPE_PROGRAM);

        // Delete shaders as they're now linked into our program
        glDeleteShader(vertex);
        glDeleteShader(fragment);
    }

    void Shader::load(const char *vertexPath, const char *fragmentPath) {
        std::string vCode, fCode;
        std::ifstream vifs, fifs;

        // Ensure that ifstream can throw exceptions
        vifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        // Load files
        try {
            vifs.open(vertexPath);
            fifs.open(fragmentPath);

            // Read files buffer into stream
            std::ostringstream vss, fss;
            vss << vifs.rdbuf();
            fss << fifs.rdbuf();

            // Save stream contentn into strings
            vCode = vss.str();
            fCode = fss.str();

            vifs.close();
            fifs.close();
        } catch (std::ifstream::failure &e) {
            std::cout << "Error - SHADER_LOAD could not load file: " << std::endl;
            std::cout << e.what() << std::endl;
        }

        // Initialize loaded shaders
        const char *c_vCode = vCode.c_str();
        const char *c_fCode = fCode.c_str();
        init(c_vCode, c_fCode);
    }

    Shader::Shader(const char *vertexPath, const char *fragmentPath) {
        load(vertexPath, fragmentPath);
    }

    void Shader::use() {
        glUseProgram(id);
    }

    void Shader::setMat4(const std::string &name, const glm::mat4 &mat) const {
        glUniformMatrix4fv(glGetUniformLocation(id, name.c_str()), 1, GL_FALSE, &mat[0][0]);
    }
}