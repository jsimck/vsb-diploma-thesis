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
        templates[key].emplace_back(&t);
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

HashTable HashTable::load(cv::FileNode &node, std::vector<Template> &templates) {
    HashTable table;

    // Load triplet triplet
    cv::FileNode tripletNode = node["triplet"];
    tripletNode["p1"] >> table.triplet.p1;
    tripletNode["p2"] >> table.triplet.p2;
    tripletNode["c"] >> table.triplet.c;

    // Load bin ranges
    node["binRanges"] >> table.binRanges;

    // Save Templates
    cv::FileNode data = node["data"];
    for (auto &&row : data) {
        // Load key
        HashKey key;
        cv::FileNode keyNode = row["key"];

        keyNode["d1"] >> key.d1;
        keyNode["d2"] >> key.d2;
        keyNode["n1"] >> key.n1;
        keyNode["n2"] >> key.n2;
        keyNode["n3"] >> key.n3;

        // Load templates
        int id = 0;
        cv::FileNode templatesNode = row["templates"];

        for (auto &&tplId : templatesNode) {
            tplId >> id;

            // Loop through existing templates and save pointers to maching ids
            for (auto &tpl : templates) {
                if (id == tpl.id) {
                    // Check if key exists, if not initialize it
                    if (table.templates.find(key) == table.templates.end()) {
                        std::vector<Template *> hashTemplates;
                        table.templates[key] = hashTemplates;
                    }

                    // Push to table
                    table.templates[key].push_back(&tpl);
                    break;
                }
            }
        }
    }

    return table;
}

void HashTable::save(cv::FileStorage &fsw) {
    // Save triplet
    fsw << "{";
    fsw << "triplet" << "{";
    fsw << "p1" << triplet.p1;
    fsw << "c" << triplet.c;
    fsw << "p2" << triplet.p2;
    fsw << "}";

    // Save bin ranges
    fsw << "binRanges" << binRanges;

    // Save Templates
    fsw << "data" << "[";
    for (auto &tableRow : templates) {
        // Save key
        fsw << "{";
        fsw << "key" << "{";
        fsw << "d1" << tableRow.first.d1;
        fsw << "d2" << tableRow.first.d2;
        fsw << "n1" << tableRow.first.n1;
        fsw << "n2" << tableRow.first.n2;
        fsw << "n3" << tableRow.first.n3;
        fsw << "}";

        // Save template IDS
        fsw << "templates" << "[";
        for (auto &tpl : tableRow.second) {
            fsw << tpl->id;
        }
        fsw << "]";
        fsw << "}";
    }
    fsw << "]";

    fsw << "}";
}
