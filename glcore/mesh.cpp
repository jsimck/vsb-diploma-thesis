#include "mesh.h"

namespace tless {
    void Mesh::load(const std::string &plyFile) {
        try {
            // Read the file and create a std::istringstream suitable
            // for the lib -- tinyply does not perform any file i/o.
            std::ifstream ss(plyFile, std::ios::binary);

            if (ss.fail()) {
                throw std::runtime_error("failed to open " + plyFile);
            }

            // Parse header
            tinyply::PlyFile file;
            file.parse_header(ss);
            std::shared_ptr<tinyply::PlyData> plyVertices, plyNormals, plyFaces;

            // Extract header information
            try { plyVertices = file.request_properties_from_element("vertex", {"x", "y", "z"}); }
            catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

            try { plyNormals = file.request_properties_from_element("vertex", {"nx", "ny", "nz"}); }
            catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

            try { plyFaces = file.request_properties_from_element("face", {"vertex_indices"}); }
            catch (const std::exception &e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

            file.read(ss);

            // Reserve array size to the size of the buffer
            std::vector<glm::vec3> _normals(plyVertices->count);
            std::vector<glm::vec3> _vertices(plyVertices->count);
            std::vector<glm::ivec3> _indices(plyFaces->count);

            // Copy data from buffer to local arrays
            std::memcpy(_indices.data(), plyFaces->buffer.get(), plyFaces->buffer.size_bytes());
            std::memcpy(_vertices.data(), plyVertices->buffer.get(), plyVertices->buffer.size_bytes());
            std::memcpy(_normals.data(), plyNormals->buffer.get(), plyNormals->buffer.size_bytes());

            // Save vertices and normals to local arrays
            for (int i = 0; i < _vertices.size(); ++i) {
                vertices.emplace_back(Vertex(_vertices[i], _normals[i]));
            }

            // Save incides to local arrays
            for (auto &ind : _indices) {
                indices.push_back(static_cast<unsigned int &&>(ind.x));
                indices.push_back(static_cast<unsigned int &&>(ind.y));
                indices.push_back(static_cast<unsigned int &&>(ind.z));
            }
        } catch (const std::exception &e) {
            std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
        }

        // Init Mesh
        init();
    }

    void Mesh::init()  {
        // Create buffers/arrays
        glGenVertexArrays(1, &id);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        // Load data into vertex buffers
        glBindVertexArray(id);
        // Bind vertex positions and normals
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        // Bind face indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Set the vertex attribute pointers
        // Vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) 0);
        // Vertex normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) offsetof(Vertex, normal));

        // Unbind current id
        glBindVertexArray(0);
    }

    Mesh::Mesh(const std::string &plyFile) {
        load(plyFile);
    }

    void Mesh::draw() const {
        glBindVertexArray(id);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, (void*) 0);
        glBindVertexArray(0);
    }

    std::ostream &tless::operator<<(std::ostream &os, const Mesh &mesh) {
        os << "VBO: " << mesh.VBO
           << " EBO: " << mesh.EBO
           << " id: " << mesh.id << std::endl;

        for (int i = 0; i < mesh.vertices.size(); ++i) {
            os << "V: " << mesh.vertices[i] <<std::endl;
        }

        for (int i = 0; i < mesh.indices.size(); i += 3) {
            os << "V: (" << mesh.vertices[i] << ", " << mesh.vertices[i + 1] << ", " << mesh.vertices[i + 2] << ")" << std::endl;
        }

        return os;
    }

    Mesh::~Mesh() {
        glDeleteBuffers(1, &VBO);
        glDeleteBuffers(1, &EBO);
        glDeleteVertexArrays(1, &id);
    }
}
