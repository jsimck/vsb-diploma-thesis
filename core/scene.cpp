#include "scene.h"

namespace tless {
    std::ostream &operator<<(std::ostream &os, const Scene &scene) {
        os << "Scene id: " << scene.id
           << "Pyramid levels: " << scene.pyramid.size() << std::endl;

        return os;
    }
}