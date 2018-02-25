#ifndef VSB_SEMESTRAL_PROJECT_HASHKEY_H
#define VSB_SEMESTRAL_PROJECT_HASHKEY_H

#include <ostream>
#include <opencv2/core/hal/interface.h>
#include <boost/functional/hash.hpp>

namespace tless {
    /**
     * @brief Custom hash key used in hash tables to quickly identify set of valid candidates for each Window.
     *
     * Custom hash key used to quickly identify valid candidates for each Window.
     * There can be max 12800 different hash keys, consisting of (d1, d2, n1, n2, n3) where:
     *   - d1, d2 are quantized relative depths (5 bins) (p1 - c, p2 - c)
     *   - n1, n2, n3 are quantized surface normals (8 bins) at (p1, p2, c)
     * this gives 5 * 5 * 8 * 8 * 8 = 12800 possible different keys.
     */
    struct HashKey {
    public:
        uchar d1 = 0, d2 = 0; //!< d1, d2 relative depths, quantization into 5 bins
        uchar n1 = 0, n2 = 0, n3 = 0; //!< n1, n2, n3 surface normals, quantized into 8 discrete values

        HashKey() = default;
        HashKey(uchar d1, uchar d2, uchar n1, uchar n2, uchar n3) : d1(d1), d2(d2), n1(n1), n2(n2), n3(n3) {}

        /**
         * @brief Returns true if key is not empty e.g. none of the key values are equal to 0
         *
         * @return false/true whether key is empty or not
         */
        bool empty();

        bool operator==(const HashKey &rhs) const;
        bool operator!=(const HashKey &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const HashKey &key);
    };

    struct HashKeyHasher {
        std::size_t operator()(const HashKey &k) const {
            std::size_t seed = 0;

            boost::hash_combine(seed, boost::hash_value(k.d1));
            boost::hash_combine(seed, boost::hash_value(k.d2));
            boost::hash_combine(seed, boost::hash_value(k.n1));
            boost::hash_combine(seed, boost::hash_value(k.n2));
            boost::hash_combine(seed, boost::hash_value(k.n3));

            return seed;
        }
    };
}

#endif
