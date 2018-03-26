#ifndef VSB_SEMESTRAL_PROJECT_HASHTABLE_H
#define VSB_SEMESTRAL_PROJECT_HASHTABLE_H

#include "triplet.h"
#include "hash_key.h"
#include "template.h"
#include <memory>
#include <ostream>
#include <utility>

namespace tless {
    /**
     * @brief Represents 1 hash table identified by unique triplet.
     *
     * Each hash table is then filled with set of valid candidates, based
     * on the custom hash key, that's formed on template quantized values.
     */
    class HashTable {
    public:
        size_t size = 0;  //!< Size of hash table (in terms of number of templates)
        Triplet triplet;
        std::vector<cv::Range> binRanges;
        std::vector<std::vector<Template*>> templates{18944};

        HashTable() = default;
        HashTable(Triplet triplet) : triplet(triplet) {}

        /**
         * @brief Loads hash table from trained classifier.yml file.
         *
         * @param[in] node      File node identifying hash table in classifier.yml file
         * @param[in] templates Templates from dataset, these are used to assign correct pointers for each hash key
         *                      (comparison is done based on matching ids)
         * @return              Parsed hash table, with all assigned template pointers
         */
        static HashTable load(cv::FileNode &node, std::vector<Template> &templates);

        /**
         * @brief Use when pushing new templates to hash table.
         *
         * This function makes sure that when you push new template to the hash table at specific key, it either
         * initializes that key and pushes the template or checks, if the template is not already present at given key
         * (checks for uniques), in that case no templates are pushed. It also increases the size counter, in case new
         * template is pushed to the table so to retain validity for the size of the table, it's crucial to only use
         * this function when putting new objects to hash table.
         *
         * @param[in] key  HashKey identifying place where to push new template t
         * @param[in] t    Template to push to hash table at specified key
         * @return         True/false whether template was pushed to the table or not
         */
        void pushUnique(const HashKey &key, Template &t);

        bool operator<(const HashTable &rhs) const;
        bool operator>(const HashTable &rhs) const;
        bool operator<=(const HashTable &rhs) const;
        bool operator>=(const HashTable &rhs) const;

        friend cv::FileStorage &operator<<(cv::FileStorage &fs, const HashTable &crit);
        friend std::ostream &operator<<(std::ostream &os, const HashTable &table);
    };
}

#endif
