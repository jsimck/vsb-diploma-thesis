#ifndef VSB_SEMESTRAL_PROJECT_HASHTABLE_H
#define VSB_SEMESTRAL_PROJECT_HASHTABLE_H

#include "triplet.h"
#include "hash_key.h"
#include "template.h"
#include <unordered_map>
#include <ostream>

/**
 * struct HashTable
 *
 * Hash table used to store template candidates with discretizied values into
 * coresponding bins, forming hash key of (d1, d2, n1, n2, n3). Each hash table is
 * represented with one triplet used to train and separate all templates into different
 * hash keys.
 */
struct HashTable {
public:
    Triplet triplet;
    std::vector<cv::Range> binRanges;
    std::unordered_map<HashKey, std::vector<Template *>, HashKeyHasher> templates;

    // Constructors
    HashTable() {}
    HashTable(const Triplet &triplet) : triplet(triplet) {}

    // Methods
    void pushUnique(const HashKey &key, Template &t);

    // Operators
    friend std::ostream &operator<<(std::ostream &os, const HashTable &table);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHTABLE_H
