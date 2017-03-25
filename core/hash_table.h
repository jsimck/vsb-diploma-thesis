#ifndef VSB_SEMESTRAL_PROJECT_HASHTABLE_H
#define VSB_SEMESTRAL_PROJECT_HASHTABLE_H

#include "triplet.h"
#include "hash_key.h"
#include "template.h"
#include <unordered_map>
#include <ostream>

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
