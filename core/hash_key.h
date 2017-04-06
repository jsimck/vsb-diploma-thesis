#ifndef VSB_SEMESTRAL_PROJECT_HASHKEY_H
#define VSB_SEMESTRAL_PROJECT_HASHKEY_H

#include <ostream>

/**
 * struct HashKey
 *
 * Custom hash key used in hash table to separate trained images into
 * it's own bins. There can be max 12800 different hash keyes, consisting of
 * (d1, d2, n1, n2, n3) where d1, d2 are depth relative depths (p2 - p1, p3 - p1)
 * and n1, n2, n3 quantizied surface normals (into 8 bins) at each point of learned triplets
 * this gives 5 * 5 * 8 * 8 * 8 = 12800 possible different keys, where each key contains
 * list of templates corresponding to discretizied values of hash key.
 */
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

    // Constructors
    HashKey(int d1, int d2, int n1, int n2, int n3) : d1(d1), d2(d2), n1(n1), n2(n2), n3(n3) {}

    // Operators
    bool operator==(const HashKey &rhs) const;
    bool operator!=(const HashKey &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const HashKey &key);
};

struct HashKeyHasher {
    std::size_t operator()(const HashKey& k) const {
        std::hash<int> h;
        return h(h(k.d1) ^ h(k.d2) ^ h(k.n1) ^ h(k.n2) ^ h(k.n3));
    }
};

#endif //VSB_SEMESTRAL_PROJECT_HASHKEY_H
