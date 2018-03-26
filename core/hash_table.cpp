#include "hash_table.h"

namespace tless {
    void HashTable::pushUnique(const HashKey &key, Template &t) {
        // Check if key exists, if not initialize it
        auto& vec = templates[key.hash()];

        if (std::find_if(vec.begin(), vec.end(), [&t](const Template *tt) { return t == *tt; }) == vec.end()) {
            vec.push_back(&t);
            size++;
        }
    }

    std::ostream &operator<<(std::ostream &os, const HashTable &table) {
        os << "Size: " << table.size << std::endl;
        os << "Triplet " << table.triplet << std::endl;

        os << "Bin ranges:" << std::endl;
        for (size_t iSize = table.binRanges.size(), i = 0; i < iSize; i++) {
            os << "  |_ " << i << ". <" << table.binRanges[i].start << ", " << table.binRanges[i].end
               << (i + 1 == table.binRanges.size() ? ">" : ")") << std::endl;
        }

        os << "Table contents: (d1, d2, n1, n2, n3)" << std::endl;
        for (int j = 0; j < table.templates.size(); ++j) {
            if (table.templates[j].empty()) {
                continue;
            }

            os << "  |_ " << HashKey::unhash(j) << " : (";
            for (const auto &item : table.templates[j]) {
                os << item->id << ", ";
            }
            os << ")" << std::endl;
        }

        return os;
    }

    HashTable HashTable::load(cv::FileNode &node, std::vector<Template> &templates) {
        HashTable table;

        int size;
        node["size"] >> size;
        table.size = static_cast<size_t>(size);
        node["binRanges"] >> table.binRanges;

        cv::FileNode tripletNode = node["triplet"];
        tripletNode["p1"] >> table.triplet.p1;
        tripletNode["p2"] >> table.triplet.p2;
        tripletNode["c"] >> table.triplet.c;

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
                table.templates[key.hash()].reserve(templatesNode.size());
                tplId >> id;

                // Loop through existing templates and save pointers to matching ids
                for (auto &t : templates) {
                    if (id == t.id) {
                        table.templates[key.hash()].push_back(&t);
                    }
                }
            }
        }

        return table;
    }

    bool HashTable::operator<(const HashTable &rhs) const {
        return size < rhs.size;
    }

    bool HashTable::operator>(const HashTable &rhs) const {
        return rhs < *this;
    }

    bool HashTable::operator<=(const HashTable &rhs) const {
        return !(rhs < *this);
    }

    bool HashTable::operator>=(const HashTable &rhs) const {
        return !(*this < rhs);
    }

    cv::FileStorage &operator<<(cv::FileStorage &fs, const HashTable &table) {
        // Save triplet
        fs << "{";
        fs << "size" << static_cast<int>(table.size);
        fs << "binRanges" << table.binRanges;
        fs << "triplet" << "{";
        fs << "p1" << table.triplet.p1;
        fs << "c" << table.triplet.c;
        fs << "p2" << table.triplet.p2;
        fs << "}";

        // Save Templates
        fs << "data" << "[";
        for (int i = 0; i < table.templates.size(); ++i) {
            HashKey key = HashKey::unhash(i);

            // Save key
            fs << "{";
            fs << "key" << "{";
            fs << "d1" << key.d1;
            fs << "d2" << key.d2;
            fs << "n1" << key.n1;
            fs << "n2" << key.n2;
            fs << "n3" << key.n3;
            fs << "}";

            // Save template IDS
            fs << "templates" << "[";
            for (auto &t : table.templates[i]) {
                fs << t->id;
            }
            fs << "]";
            fs << "}";
        }
        fs << "]";
        fs << "}";

        return fs;
    }
}