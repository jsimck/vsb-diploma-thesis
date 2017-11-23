#ifndef VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H
#define VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H

#include <memory>
#include <utility>
#include "template.h"

class HashTableCandidate {
public:
    int votes = 0;
    std::shared_ptr<Template> candidate;

    // Constructor
    HashTableCandidate(std::shared_ptr<Template> candidate = nullptr) : candidate(candidate) {}

    // Methods
    void vote();

    // Operators
    bool operator<(const HashTableCandidate &rhs) const;
    bool operator>(const HashTableCandidate &rhs) const;
    bool operator<=(const HashTableCandidate &rhs) const;
    bool operator>=(const HashTableCandidate &rhs) const;
};

#endif
