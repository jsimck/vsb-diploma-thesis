#ifndef VSB_SEMESTRAL_PROJECT_HASHKEY_H
#define VSB_SEMESTRAL_PROJECT_HASHKEY_H

#include <ostream>
#include <boost/functional/hash.hpp>

struct HashKey {
public:
    // Key values
    // d1, d2 relative depths are quantization into 5 bins each [index 0 - 4]
    // n1, n2, n3 surface normals are quantized into 8 discrete values each [index 0 - 7]
    union {
        struct {
            int d1, d2;
            int n1, n2, n3;
        };

        int key[5];
    };

    HashKey(int d1, int d2, int n1, int n2, int n3) : d1(d1), d2(d2), n1(n1), n2(n2), n3(n3) {}

    bool operator==(const HashKey &rhs) const;
    bool operator!=(const HashKey &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const HashKey &key);
};

struct HashKeyHasher {
    std::size_t operator()(const HashKey& k) const {
        std::size_t seed = 0;

        boost::hash_combine(seed, k.d1);
        boost::hash_combine(seed, k.d2);
        boost::hash_combine(seed, k.n1);
        boost::hash_combine(seed, k.n2);
        boost::hash_combine(seed, k.n3);

        return seed;
    }
};

#endif //VSB_SEMESTRAL_PROJECT_HASHKEY_H
