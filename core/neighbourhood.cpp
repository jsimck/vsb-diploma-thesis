#include "neighbourhood.h"

Neighbourhood::Neighbourhood(int width, int height) {
    this->width = width;
    this->height = height;
    offsetX = width / 2;
    offsetY = height / 2;
}
