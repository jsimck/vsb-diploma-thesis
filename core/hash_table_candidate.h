#ifndef VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H
#define VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H

#include "template.h"

struct HashTableCandidate {
public:
    int votes;
    Template *candidate;

    // Constructor
    HashTableCandidate(Template *candidate = nullptr) : candidate(candidate), votes(0) {}

    // Methods
    void vote();

    // Operators
    bool operator<(const HashTableCandidate &rhs) const;
    bool operator>(const HashTableCandidate &rhs) const;
    bool operator<=(const HashTableCandidate &rhs) const;
    bool operator>=(const HashTableCandidate &rhs) const;
};

#endif //VSB_SEMESTRAL_PROJECT_HASH_TABLE_CANDIDATE_H
