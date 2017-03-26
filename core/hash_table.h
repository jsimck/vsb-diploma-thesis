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
 * Hash table used to store trained templates with discretizied values into
 * coresponding bins, forming hash key of (d1, d2, n1, n2, n3)
 */
struct HashTable {
public:
    Triplet triplet;
    std::unordered_map<HashKey, std::vector<Template>, HashKeyHasher> templates;

    // Constructors
    HashTable() {}

    // Operators
    friend std::ostream &operator<<(std::ostream &os, const HashTable &table);
};

#endif //VSB_SEMESTRAL_PROJECT_HASHTABLE_H
