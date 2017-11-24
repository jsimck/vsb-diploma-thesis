#include "hash_table.h"

namespace tless {
    void HashTable::pushUnique(const HashKey &key, Template &t) {
        // Check if key exists, if not initialize it
        if (templates.find(key) == templates.end()) {
            std::vector<std::shared_ptr<Template>> hashTemplates;
            templates[key] = hashTemplates;
        }

        // Check for duplicates and push unique
        auto found = std::find_if(templates[key].begin(), templates[key].end(),
                                  [&t](const std::shared_ptr<Template> &tt) { return t == *tt; });
        if (found == templates[key].end()) {
            templates[key].emplace_back(&t);
        }
    }

    std::ostream &operator<<(std::ostream &os, const HashTable &table) {
        os << "Triplet " << table.triplet << std::endl;

        os << "Bin ranges:" << std::endl;
        for (size_t iSize = table.binRanges.size(), i = 0; i < iSize; i++) {
            os << "  |_ " << i << ". <" << table.binRanges[i].start << ", " << table.binRanges[i].end
               << (i + 1 == table.binRanges.size() ? ">" : ")") << std::endl;
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
        cv::FileNode tripletNode = node["triplet"];
        tripletNode["p1"] >> table.triplet.p1;
        tripletNode["p2"] >> table.triplet.p2;
        tripletNode["c"] >> table.triplet.c;
        node["binRanges"] >> table.binRanges;

        // Load Templates
        cv::FileNode data = node["data"];
        for (auto &&row : data) {
            HashKey key;
            cv::FileNode keyNode = row["key"];

            keyNode["d1"] >> key.d1;
            keyNode["d2"] >> key.d2;
            keyNode["n1"] >> key.n1;
            keyNode["n2"] >> key.n2;
            keyNode["n3"] >> key.n3;

            int id = 0;
            cv::FileNode templatesNode = row["templates"];

            for (auto &&tplId : templatesNode) {
                tplId >> id;

                // Loop through existing templates and save pointers to matching ids
                for (auto &tpl : templates) {
                    if (id == tpl.id) {
                        // Check if key exists, if not initialize it
                        if (table.templates.find(key) == table.templates.end()) {
                            std::vector<std::shared_ptr<Template>> hashTemplates;
                            table.templates[key] = hashTemplates;
                        }

                        table.templates[key].emplace_back(&tpl);
                        break;
                    }
                }
            }
        }

        return table;
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const HashTable &table) {
        // Save triplet
        fs << "{";
        fs << "triplet" << "{";
        fs << "p1" << table.triplet.p1;
        fs << "c" << table.triplet.c;
        fs << "p2" << table.triplet.p2;
        fs << "}";
        fs << "binRanges" << table.binRanges;

        // Save Templates
        fs << "data" << "[";
        for (auto &tableRow : table.templates) {
            // Save key
            fs << "{";
            fs << "key" << "{";
            fs << "d1" << tableRow.first.d1;
            fs << "d2" << tableRow.first.d2;
            fs << "n1" << tableRow.first.n1;
            fs << "n2" << tableRow.first.n2;
            fs << "n3" << tableRow.first.n3;
            fs << "}";

            // Save template IDS
            fs << "templates" << "[";
            for (auto &tpl : tableRow.second) {
                fs << static_cast<int>(tpl->id);
            }
            fs << "]";
            fs << "}";
        }
        fs << "]";
        fs << "}";

        return fs;
    }
}