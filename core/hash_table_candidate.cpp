#include "hash_table_candidate.h"

namespace tless {
    void HashTableCandidate::vote() {
        votes++;
    }

    bool HashTableCandidate::operator<(const HashTableCandidate &rhs) const {
        return votes < rhs.votes;
    }

    bool HashTableCandidate::operator>(const HashTableCandidate &rhs) const {
        return rhs < *this;
    }

    bool HashTableCandidate::operator<=(const HashTableCandidate &rhs) const {
        return !(rhs < *this);
    }

    bool HashTableCandidate::operator>=(const HashTableCandidate &rhs) const {
        return !(*this < rhs);
    }
}