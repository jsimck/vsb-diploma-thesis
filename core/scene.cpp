#include "scene.h"

std::ostream &operator<<(std::ostream &os, const Scene &scene) {
    os << "camera: " << scene.camera
       << " scale: " << scene.scale
       << " elev: " << scene.elev
       << " mode: " << scene.mode;

    return os;
}
