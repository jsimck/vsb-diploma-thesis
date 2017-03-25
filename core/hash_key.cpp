#include "hash_key.h"
#include <boost/functional/hash.hpp>

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

std::ostream &operator<<(std::ostream &os, const HashKey &key) {
    os << "(" << key.d1 << ", " << key.d2 << ", " << key.n1 << ", " << key.n2 << ", " << key.n3 << ")";
    return os;
}