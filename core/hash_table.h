#ifndef VSB_SEMESTRAL_PROJECT_HASHTABLE_H
#define VSB_SEMESTRAL_PROJECT_HASHTABLE_H

#include "triplet.h"
#include "hash_key.h"
#include "template.h"
#include <unordered_map>

struct HashTable {
public:
    std::vector<Triplet> triplets;
    std::unordered_map<HashKey, std::vector<Template>, HashKeyHasher> templates;
};

#endif //VSB_SEMESTRAL_PROJECT_HASHTABLE_H
