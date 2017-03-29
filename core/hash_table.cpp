#include "hash_table.h"

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
