/* The same timing harness as harness.cpp, for the C port: loads the pack's vocab.json/merges.txt,
 * encodes corpus.bin (uint32 length + bytes per item), writes ids, prints load/encode timings.

 usage: harness_strata <pack_dir> <corpus.bin> <ids.out> [cold|warm] [pieces]
*/
#include "strata_bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double ms(struct timespec a, struct timespec b)
{
    return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) / 1e6;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s <pack_dir> <corpus.bin> <ids.out> [cold|warm] [pieces]\n", argv[0]); return 2; }
    int warm = argc > 4 && strcmp(argv[4], "warm") == 0;
    int dump = argc > 5;

    char vocab[1024], merges[1024];
    snprintf(vocab, sizeof(vocab), "%s/vocab.json", argv[1]);
    snprintf(merges, sizeof(merges), "%s/merges.txt", argv[1]);

    char err[256] = {0};
    sb_tok *t = NULL;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = sb_load(&t, vocab, merges, err, sizeof(err));
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc != 0) { fprintf(stderr, "sb_load: %s\n", err); return 1; }
    printf("load %.1f ms\n", ms(t0, t1));

    FILE *f = fopen(argv[2], "rb");
    if (f == NULL) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    uint32_t *lens = NULL;
    char *items = NULL;
    size_t nitems = 0, total = 0;
    for (;;) {
        uint32_t len;
        if (fread(&len, 4, 1, f) != 1) { break; }
        items = realloc(items, total + len);
        if (fread(items + total, 1, len, f) != len) { fprintf(stderr, "truncated corpus\n"); return 1; }
        lens = realloc(lens, (nitems + 1) * sizeof(uint32_t));
        lens[nitems++] = len;
        total += len;
    }
    fclose(f);

    uint32_t *ids = malloc((total + 16) * sizeof(uint32_t));
    FILE *pieces = NULL;
    if (dump) {
        pieces = fopen(argv[5], "wb");
        if (pieces == NULL) { fprintf(stderr, "cannot open %s\n", argv[5]); return 1; }
    }
    size_t off = 0, filled = 0;
    struct timespec e0, e1;
    clock_gettime(CLOCK_MONOTONIC, &e0);
    for (size_t i = 0; i < nitems; i++) {
        int got = sb_encode(t, items + off, lens[i], ids + filled, total + 16 - filled, warm);
        if (got < 0) { fprintf(stderr, "encode failed on item %zu\n", i); return 1; }
        filled += (size_t)got;
        if (pieces) { sb_split_dump(t, items + off, lens[i], pieces); }
        off += lens[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &e1);

    FILE *out = fopen(argv[3], "wb");
    if (out == NULL) { fprintf(stderr, "cannot open %s\n", argv[3]); return 1; }
    fwrite(ids, sizeof(uint32_t), filled, out);
    fclose(out);
    if (pieces) { fclose(pieces); }
    printf("encode %zu items %zu ids %.1f ms (%s) %.1f ns/byte %.1f MB/s\n",
           nitems, filled, ms(e0, e1), warm ? "warm" : "cold",
           ms(e0, e1) * 1e6 / (double)total, (double)total / (1024.0 * 1024.0 * (ms(e0, e1) / 1000.0)));
    sb_free(t);
    free(ids); free(lens); free(items);
    return 0;
}
