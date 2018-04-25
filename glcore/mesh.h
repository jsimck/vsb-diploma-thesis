#ifndef VSB_SEMESTRAL_PROJECT_MESH_H
#define VSB_SEMESTRAL_PROJECT_MESH_H

#include <iostream>
#include <fstream>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include "../libs/tinyply.h"

namespace tless {
    /**
     * @brief Helper class which contains pair of two 3D vectors (position, normals)
     */
    struct Vertex {
        glm::vec3 position;
        glm::vec3 normal;

        Vertex(glm::vec3 position, glm::vec3 normal) : position(position), normal(normal) {}

        friend std::ostream &operator<<(std::ostream &os, const Vertex &vertex) {
            os << "pos("
               << vertex.position.x << ", "
               << vertex.position.y << ", "
               << vertex.position.z << ") n("
               << vertex.normal.x << ", "
               << vertex.normal.y << ", "
               << vertex.normal.z << ")";

            return os;
        }
    };

    /**
     * @brief Helper class which loads mesh models from .ply files and initializes them to opengl.
     */
    class Mesh {
    private:
        GLuint VBO, EBO;

        /**
         * @brief Initializes all the VAOs, VBOs for OpenGL.
         */
        void init();

    public:
        GLuint id;
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        Mesh() = default;
        Mesh(const std::string &plyFile);

        /**
         * @brief Parses mesh from .ply file using tinyply library and calls init.
         *
         * @param[in] plyFile Path to the .ply file to parse mesh from
         */
        void load(const std::string &plyFile);

        /**
         * @brief Binds initialized VAOs and draws mesh using glDrawElements.
         */
        void draw() const;

        friend std::ostream &operator<<(std::ostream &os, const Mesh &mesh);
    };
}

#endif
