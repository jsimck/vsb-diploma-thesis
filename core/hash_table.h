#ifndef VSB_SEMESTRAL_PROJECT_HASHTABLE_H
#define VSB_SEMESTRAL_PROJECT_HASHTABLE_H

#include "triplet.h"
#include "hash_key.h"
#include "template.h"
#include <unordered_map>
#include <memory>
#include <ostream>
#include <utility>

namespace tless {
    /**
     * @brief Represents 1 hash table identified by unique triplet.
     *
     * Each hash table is then filled with set of valid candidates, based
     * on the custom hash key, that's formed on template quantized values.
     */
    class HashTable {
    public:
        Triplet triplet;
        std::vector<cv::Range> binRanges;
        std::unordered_map<HashKey, std::vector<std::shared_ptr<Template>>, HashKeyHasher> templates;

        // Constructors
        HashTable() {}
        HashTable(Triplet triplet) : triplet(triplet) {}

        // Methods
        static HashTable load(cv::FileNode &node, std::vector<Template> &templates);
        void pushUnique(const HashKey &key, Template &t);

        // Operators & friends
        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const HashTable &crit);
        friend std::ostream &operator<<(std::ostream &os, const HashTable &table);
    };
}

#endif
