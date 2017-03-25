#include "hash_key.h"

bool HashKey::operator==(const HashKey &rhs) const {
    return d1 == rhs.d1 &&
           d2 == rhs.d2 &&
           n1 == rhs.n1 &&
           n2 == rhs.n2 &&
           n3 == rhs.n3;
}

bool HashKey::operator!=(const HashKey &rhs) const {
    return !(rhs == *this);
}
