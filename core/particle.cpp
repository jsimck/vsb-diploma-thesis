#include "particle.h"
#include <glm/ext.hpp>
#include <iostream>

Particle::Particle(float tx, float ty, float tz, float rx, float ry, float rz) {
    this->tx = tx;
    this->ty = ty;
    this->tz = tz;
    this->rx = rx;
    this->ry = ry;
    this->rz = rz;

    // Init pBest
    pBest.tx = tx;
    pBest.ty = ty;
    pBest.tz = tz;
    pBest.rx = rx;
    pBest.ry = ry;
    pBest.rz = rz;
};

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
    os << "fitness: " << particle.fitness << " tx: " << particle.tx << " ty: " << particle.ty << " tz: " << particle.tz << " rx: " << particle.rx << " ry: "
       << particle.ry << " rz: " << particle.rz;
    return os;
}
