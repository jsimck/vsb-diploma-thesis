#ifndef VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H
#define VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H

#include <memory>
#include <utility>
#include "template.h"

struct HashTableCandidate {
public:
    int votes;
    std::shared_ptr<Template> candidate;

    // Constructor
    HashTableCandidate(std::shared_ptr<Template> candidate = nullptr) : votes(0), candidate(candidate) {}

    // Methods
    void vote();

    // Operators
    bool operator<(const HashTableCandidate &rhs) const;
    bool operator>(const HashTableCandidate &rhs) const;
    bool operator<=(const HashTableCandidate &rhs) const;
    bool operator>=(const HashTableCandidate &rhs) const;
};

#endif //VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H
