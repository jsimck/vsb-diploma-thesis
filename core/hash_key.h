#ifndef VSB_SEMESTRAL_PROJECT_HASHKEY_H
#define VSB_SEMESTRAL_PROJECT_HASHKEY_H


struct HashKey {
private:
    // Key values
    // d1, d2 relative depths are quantization into 5 bins each [index 0 - 4]
    // n1, n2, n3 surface normals are quantized into 8 discrete values each [index 0 - 7]
    char d1, d2;
    char n1, n2, n3;
public:
    HashKey(char d1, char d2, char n1, char n2, char n3) : d1(d1), d2(d2), n1(n1), n2(n2), n3(n3) {}

    bool operator==(const HashKey &rhs) const;
    bool operator!=(const HashKey &rhs) const;
};


#endif //VSB_SEMESTRAL_PROJECT_HASHKEY_H
