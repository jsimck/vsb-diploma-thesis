#include "particle.h"
#include <glm/ext.hpp>
#include <iostream>

glm::mat4 Particle::model()  {
    glm::mat4 m;
    glm::vec3 t(tx, ty, tz);

    // Rotate
    m = glm::rotate(m, rx, glm::vec3(1, 0, 0));
    m = glm::rotate(m, ry, glm::vec3(0, 1, 0));
    m = glm::rotate(m, rz, glm::vec3(0, 0, 1));

    // Translate
    return glm::translate(m, t);
}

std::ostream &operator<<(std::ostream &os, const Particle &particle) {
    os << "tx: " << particle.tx << " ty: " << particle.ty << " tz: " << particle.tz << " rx: " << particle.rx << " ry: "
       << particle.ry << " rz: " << particle.rz;
    return os;
};
