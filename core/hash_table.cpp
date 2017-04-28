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
    for (const auto &entry : table.templates) {
        os << entry.first << " : (";
        for (const auto &item : entry.second) {
            os << item->id << ", ";
        }
        os << ")" << std::endl;
    }
    return os;
}