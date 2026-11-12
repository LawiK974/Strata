// src/tokenizer/toks_tokenizer.cpp - the CLI for the toks-backed pack tokenizer.
//
//   strata-toks-tokenizer <pack tokenizer dir> <corpus.bin> <ids.out>
//
// corpus.bin is the length-prefixed format the bench harnesses use: per item a little-endian uint32 byte
// count then the UTF-8 bytes.  ids.out is one line of space-separated ids per item, in corpus order.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "strata/tokenizer/toks_tokenizer.hpp"

int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <pack tokenizer dir> <corpus.bin> <ids.out>\n", argv[0]);
        return 2;
    }
    strata::ToksPackTokenizer tok;
    std::string err;
    if (tok.load(argv[1], err) != 0) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    std::ofstream out(argv[3], std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot open %s\n", argv[3]);
        return 1;
    }
    uint64_t total_ids = 0;
    for (;;) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char *>(&len), sizeof(len));
        if (!in) { break; }
        std::string text(size_t(len), '\0');
        in.read(text.data(), std::streamsize(len));
        if (!in) {
            std::fprintf(stderr, "corpus item is cut short\n");
            return 1;
        }
        std::vector<uint32_t> ids;
        const int64_t n = tok.encode(text, ids);
        if (n < 0) {
            std::fprintf(stderr, "encode failed: %lld\n", static_cast<long long>(n));
            return 1;
        }
        ids.resize(size_t(n));
        for (size_t k = 0; k < ids.size(); k++) {
            if (k != 0) { out << ' '; }
            out << ids[k];
        }
        out << '\n';
        total_ids += uint64_t(n);
    }
    std::printf("ids %llu, vocab %llu\n", static_cast<unsigned long long>(total_ids),
                static_cast<unsigned long long>(tok.ids_count()));
    return 0;
}
