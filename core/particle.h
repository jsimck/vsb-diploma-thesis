#ifndef VSB_SEMESTRAL_PROJECT_PARTICLE_H
#define VSB_SEMESTRAL_PROJECT_PARTICLE_H

#include <glm/glm.hpp>
#include <ostream>

struct Particle {
public:
    union {
        struct {
            float v1, v2, v3, v4, v5, v6;
            float tx, ty, tz;
            float rx, ry, rz;
            float fitness;
        };
    };

    // Personal best
    union {
        struct {
            float v1, v2, v3, v4, v5, v6;
            float tx, ty, tz;
            float rx, ry, rz;
            float fitness;
        };
    } pBest;

    Particle() = default;
    Particle(float tx, float ty, float tz, float rx, float ry, float rz,
             float v1, float v2, float v3, float v4, float v5, float v6);

    glm::mat4 model();
    void update();
    void updatePBest();

    friend std::ostream &operator<<(std::ostream &os, const Particle &particle);
};


#endif //VSB_SEMESTRAL_PROJECT_PARTICLE_H
