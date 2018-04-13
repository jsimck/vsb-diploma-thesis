#ifndef VSB_SEMESTRAL_PROJECT_HASHKEY_H
#define VSB_SEMESTRAL_PROJECT_HASHKEY_H

#include <ostream>
#include <opencv2/core/hal/interface.h>
#include <boost/functional/hash.hpp>
#include <c++/v1/iostream>

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
    private:
        /**
         * @brief Converts key values from 8bit to 3bit representation in order to reduce hash key length
         *
         * @param[in] value uchar value to convert to 3bit decimal number (2^1 - 2^7 -> 0-7)
         * @return    returns value between 0-7 depending on the power of the on the input
         */
        inline size_t hashValue(uchar value) const;

    public:
        uchar d1 = 0, d2 = 0; //!< d1, d2 relative depths, quantization into 5 bins
        uchar n1 = 0, n2 = 0, n3 = 0; //!< n1, n2, n3 surface normals, quantized into 8 discrete values

        HashKey() = default;
        HashKey(uchar d1, uchar d2, uchar n1, uchar n2, uchar n3) : d1(d1), d2(d2), n1(n1), n2(n2), n3(n3) {}

        /**
         * @brief Returns true if key is not empty e.g. none of the key values are equal to 0.
         *
         * @return false/true whether key is empty or not
         */
        bool empty();

        /**
         * @brief Hashes this key and returns index to which it belongs in tables array
         *
         * @return Index of hash table templates array to which this key belongs
         */
        size_t hash() const;

        /**
         * @brief Returns original key from a already hashed one
         *
         * @return HashKey retrieved from already hashed value
         */
        static HashKey unhash(size_t key);

        bool operator==(const HashKey &rhs) const;
        bool operator!=(const HashKey &rhs) const;
        friend std::ostream &operator<<(std::ostream &os, const HashKey &key);
    };
}

#endif
