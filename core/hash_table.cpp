#include "hash_table.h"

void HashTable::pushUnique(const HashKey &key, Template &t) {
    // Check if key exists, if not initialize it
    if (templates.find(key) == templates.end()) {
        std::vector<Template *> hashTemplates;
        templates[key] = hashTemplates;
    }

    // Check for duplicates and push unique
    auto found = std::find_if(templates[key].begin(), templates[key].end(), [&t](const Template* tt) { return t == *tt; });
    if (found == templates[key].end()) {
        templates[key].push_back(&t);
    }
}

std::ostream &operator<<(std::ostream &os, const HashTable &table) {
    os << "Triplet " << table.triplet << std::endl;

    os << "Bin ranges:" << std::endl;
    for (size_t iSize = table.binRanges.size(), i = 0; i < iSize; i++) {
        os << "  |_ " << i << ". <" << table.binRanges[i].start << ", " << table.binRanges[i].end << (i + 1 == table.binRanges.size() ? ">" : ")") << std::endl;
    }

    os << "Table contents: (d1, d2, n1, n2, n3)" << std::endl;
    for (const auto &entry : table.templates) {
        os << "  |_ " << entry.first << " : (";
        for (const auto &item : entry.second) {
            os << item->id << ", ";
        }
        os << ")" << std::endl;
    }
    return os;
}