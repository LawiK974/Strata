// Minimal C++ harness for the toks experiment: it does nothing but load a tokenizer.json and encode a
// corpus.  The corpus is a flat binary (uint32 little-endian length, then the bytes) so this file needs no
// JSON parsing of its own, and the ids it writes are compared against Strata's Python tokenizer by compare.py.
//
// Build:  g++ -O2 -I <toks>/include harness.cpp -L <toks>/build/linux-x86_64 -ltoks -o harness
// Run:    ./harness <tokenizer.json> <corpus.bin> <ids.out> [cold|warm]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include "toks.h"

static std::vector<std::string> read_corpus(const char *path)
{
    std::vector<std::string> items;
    FILE *f = fopen(path, "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    uint32_t len;
    while (fread(&len, 4, 1, f) == 1) {
        std::string s(len, '\0');
        if (len != 0 && fread(&s[0], 1, len, f) != len) { fprintf(stderr, "short read in %s\n", path); exit(1); }
        items.push_back(std::move(s));
    }
    fclose(f);
    return items;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: harness <tokenizer.json> <corpus.bin> <ids.out> [cold|warm]\n"); return 1; }
    const bool cold = argc >= 5 && strcmp(argv[4], "cold") == 0;

    toks_ctx *ctx = NULL;
    toks_diag diag;
    memset(&diag, 0, sizeof(diag));
    toks_load_opts opts;
    memset(&opts, 0, sizeof(opts));
    opts.size = sizeof(opts);
    opts.diag = &diag;
    auto t0 = std::chrono::steady_clock::now();
    int64_t rc = toks_load(&ctx, argv[1], &opts);
    if (rc != 0) {
        fprintf(stderr, "toks_load: %lld %s\n", (long long)rc, diag.what);
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();

    toks_info info;
    memset(&info, 0, sizeof(info));
    info.size = sizeof(info);
    toks_get_info(ctx, &info);

    std::vector<std::string> items = read_corpus(argv[2]);
    uint64_t max_len = 0;
    for (const std::string &s : items) { if (s.size() > max_len) { max_len = s.size(); } }
    uint32_t scratch_flags = cold ? TOKS_SCRATCH_MEMO_MIB(0) : 0;
    uint64_t need = toks_scratch_bytes(ctx, max_len, scratch_flags);
    std::vector<uint8_t> scratch(need);
    uint32_t flags = TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS;
    std::vector<uint32_t> out(toks_encode_bound(ctx, max_len));

    FILE *ids = fopen(argv[3], "w");
    if (ids == NULL) { fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    uint64_t total_bytes = 0, n_ids = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < items.size(); i++) {
        if (cold) {                                   // every item sees an empty scratch, as a cold server would
            if (toks_scratch_init(ctx, scratch.data(), need, scratch_flags) < 0) { fprintf(stderr, "scratch_init\n"); return 1; }
        } else if (i == 0) {
            if (toks_scratch_init(ctx, scratch.data(), need, scratch_flags) < 0) { fprintf(stderr, "scratch_init\n"); return 1; }
        }
        int64_t n = toks_encode(ctx, items[i].data(), items[i].size(), flags, out.data(), out.size(), scratch.data());
        if (n < 0) { fprintf(stderr, "item %zu: toks_encode %lld\n", i, (long long)n); return 1; }
        for (int64_t k = 0; k < n; k++) { fprintf(ids, "%u ", out[k]); }
        fprintf(ids, "\n");
        total_bytes += items[i].size();
        n_ids += (uint64_t)n;
    }
    auto end = std::chrono::steady_clock::now();
    fclose(ids);

    double load_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double enc_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("toks %s, %s, ids %u, tier %u, paths %u, control_isolation %u\n",
           toks_version(), info.name, info.n_ids, info.tier, info.paths, info.control_isolation);
    printf("load %.0f ms, %zu items, %llu bytes, %llu ids\n", load_ns / 1e6, items.size(),
           (unsigned long long)total_bytes, (unsigned long long)n_ids);
    printf("encode %.0f ms (%s), %.2f ns/byte, %.1f MB/s\n", enc_ns / 1e6, cold ? "cold" : "warm",
           enc_ns / (double)total_bytes, (double)total_bytes / (enc_ns / 1e9) / (1024.0 * 1024.0));
    toks_unload(ctx);
    return 0;
}
