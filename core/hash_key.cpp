#include "hash_key.h"

namespace tless {
    size_t HashKey::hashValue(uchar value) const {
        for (int i = 0; i < 8; i++) {
            if (value & (1 << i)) {
                return static_cast<size_t>(i);
            }
        }

        return 0;
    }

    bool HashKey::empty() {
        return d1 == 0 && d2 == 0 && n1 == 0 && n2 == 0 && n3 == 0;
    }

    size_t HashKey::hash() const {
        return (hashValue(d1) << 12) | (hashValue(d2) << 9) | (hashValue(n1) << 6) | (hashValue(n2) << 3) | hashValue(n3);
    }

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

    HashKey HashKey::unhash(size_t key) {
        HashKey k;
        k.n3 = static_cast<uchar>(std::pow(2, (0b0000000000000111 & key)));
        k.n2 = static_cast<uchar>(std::pow(2, ((0b0000000000111000 & key) >> 3)));
        k.n1 = static_cast<uchar>(std::pow(2, ((0b0000000111000000 & key) >> 6)));
        k.d2 = static_cast<uchar>(std::pow(2, ((0b0000111000000000 & key) >> 9)));
        k.d1 = static_cast<uchar>(std::pow(2, ((0b0111000000000000 & key) >> 12)));

        return k;
    }
}