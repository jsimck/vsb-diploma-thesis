#include "particle.h"
#include <glm/ext.hpp>
#include <iostream>

Particle::Particle(float tx, float ty, float tz, float rx, float ry, float rz,
                   float v1, float v2, float v3, float v4, float v5, float v6) {
    this->tx = tx;
    this->ty = ty;
    this->tz = tz;
    this->rx = rx;
    this->ry = ry;
    this->rz = rz;
    this->v1 = v1;
    this->v2 = v2;
    this->v3 = v3;
    this->v4 = v4;
    this->v5 = v5;
    this->v6 = v6;
    this->fitness = fitness;

    // Init pBest
    updatePBest();
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

void Particle::update() {
    // Update with given velocity
    tx = v1 + tx;
    ty = v2 + ty;
    tz = v3 + tz;
    rx = v4 + rx;
    ry = v5 + ry;
    rz = v6 + rz;
}

void Particle::updatePBest() {
    pBest.tx = tx;
    pBest.ty = ty;
    pBest.tz = tz;
    pBest.rx = rx;
    pBest.ry = ry;
    pBest.rz = rz;
    pBest.v1 = v1;
    pBest.v2 = v2;
    pBest.v3 = v3;
    pBest.v4 = v4;
    pBest.v5 = v5;
    pBest.v6 = v6;
    pBest.fitness = fitness;
}

std::ostream &operator<<(std::ostream &os, const Particle &particle) {
    os << "fitness: " << particle.fitness << " tx: " << particle.tx << " ty: " << particle.ty << " tz: " << particle.tz << " rx: " << particle.rx << " ry: "
       << particle.ry << " rz: " << particle.rz;
    return os;
}