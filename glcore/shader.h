#ifndef VSB_SEMESTRAL_PROJECT_SHADER_H
#define VSB_SEMESTRAL_PROJECT_SHADER_H

#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>

namespace tless {
    class Shader {
    private:
        /**
         * @brief Utility function to check for compile errors when compiling shader
         *
         * @param[in] shader ID of compiled shader
         * @param[in] type   Shader type (TYPE_VERTEX / TYPE_FRAGMENT / TYPE_PROGRAM)
         */
        void checkCompileErrors(GLuint shader, int type);

        /**
         * @brief Utility function which compiles shader sources passed in the input params
         *
         * @param[in] vertexShader   Vertex shader code
         * @param[in] fragmentShader Fragment shader code
         */
        void init(const char *vertexShader, const char *fragmentShader);

    public:
        GLuint id;
        static const int TYPE_VERTEX = 0, TYPE_FRAGMENT = 1, TYPE_PROGRAM = 2;

        Shader() = default;
        Shader(const char *vertexPath, const char *fragmentPath);

        /**
         * @brief Utility function which loads and compiles vertex and fragment shaders from given path
         *
         * @param[in] vertexPath   Path to vertex shader file
         * @param[in] fragmentPath Path to fragment shader file
         */
        void load(const char *vertexPath, const char *fragmentPath);

        /**
         * @brief Call when you want to use this shader for rendering
         */
        void use() const;

        /**
         * @brief Utility function to quickly set mat4 uniform on this shader
         *
         * @param[in] name Uniform name on glsl shader
         * @param[in] mat  Source matrix to set on this shader
         */
        void setMat4(const std::string &name, const glm::mat4 &mat) const;
    };
};

#endif
