#include "hash_key.h"

namespace tless {
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
        os << "("
           << static_cast<int>(key.d1) << ", "
           << static_cast<int>(key.d2) << ", "
           << static_cast<int>(key.n1) << ", "
           << static_cast<int>(key.n2) << ", "
           << static_cast<int>(key.n3)
           << ")";

        return os;
    }
}