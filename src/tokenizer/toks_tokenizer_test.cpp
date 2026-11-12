// src/tokenizer/toks_tokenizer_test.cpp - the toks-backed pack tokenizer, checked against a real pack.
//
//   toks_tokenizer_test <pack tokenizer dir>
//
// What it asserts: the pack loads, toks' id count equals the pack's vocab_size, every non-empty item
// encodes to ids inside that range, the same text encodes to the same ids twice (the memo cache does not
// change the answer), and decode(encode(s)) == s byte-for-byte for multi-byte UTF-8, combining marks,
// emoji, whitespace, C0 controls and a long ASCII run.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "strata/tokenizer/toks_tokenizer.hpp"

namespace {

std::vector<std::string> items() {
    return {
        "hello world",
        "",
        "  spaces\tand\nnewlines\r\n",
        "\x01\x02\x1f control bytes",
        "café naïve — Berlin ✓",
        "日本語のテキストと中文",
        "emoji 🙂🚀🧑‍💻 and Z̈́x́ combining marks",
        "e\u0301combining\u0301mark",
        "a very long ASCII run: " + std::string(4000, 'x'),
        "\xef\xbf\xbd replacement byte",
        "{\"json\": \"quotes and \\backslash\"}",
    };
}

bool check(const char *what, bool ok) {
    std::printf("%-46s %s\n", what, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <pack tokenizer dir>\n", argv[0]);
        return 2;
    }
    strata::ToksPackTokenizer tok;
    std::string err;
    if (tok.load(argv[1], err) != 0) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    const uint64_t n_ids = tok.ids_count();
    bool ok = check("vocab size from toks", n_ids > 0 && n_ids == tok.vocab_size);
    ok = check("merges count from meta", tok.n_merges > 0) && ok;
    ok = check("pre_pattern taken from pack", !tok.pre_pattern.empty()) && ok;

    for (const std::string &s : items()) {
        std::vector<uint32_t> ids;
        const int64_t n = tok.encode(s, ids);
        ok = check("encode returns ids for a non-empty item", !s.empty() ? n > 0 : n >= 0) && ok;
        ids.resize(size_t(n < 0 ? 0 : n));
        bool in_range = true;
        for (uint32_t id : ids) { if (id >= n_ids) { in_range = false; } }
        ok = check("every id is inside the vocab", in_range) && ok;

        std::vector<uint32_t> again;
        const int64_t n2 = tok.encode(s, again);
        again.resize(size_t(n2 < 0 ? 0 : n2));
        ok = check("encoding twice gives the same ids", n2 == n && again == ids) && ok;

        std::string back;
        const int64_t written = tok.decode(ids, back);
        ok = check("decode(encode(s)) == s", written >= 0 && back == s) && ok;
    }

    std::printf("%s\n", ok ? "all checks passed" : "checks FAILED");
    return ok ? 0 : 1;
}
