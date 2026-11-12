#ifndef STRATA_BPE_H
#define STRATA_BPE_H
/* A C port of tools/strata_tokenizer.py's algorithm (byte-level BPE + the qwen35 pre-tokenizer), for
 * the bench/toks-tokenizer experiment only: it reads the pack's vocab.json / merges.txt, encodes plain
 * text with no special tokens, and is checked against the Python by compare.py.
 */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

typedef struct sb_tok sb_tok;

/* Load <pack>/vocab.json (token -> id) and <pack>/merges.txt (rank = line index).  Returns 0, or a
 * negative code with a message in err. */
int sb_load(sb_tok **out, const char *vocab_json, const char *merges_txt, char *err, size_t errlen);
void sb_free(sb_tok *t);

/* Encode plain text, exactly as Tokenizer._encode_plain: split, byte-map, BPE, vocab ids.
 * use_cache keeps the piece cache across calls (warm); 0 clears it per call (cold).  Returns the
 * number of ids written, or negative. */
int sb_encode(sb_tok *t, const char *text, size_t len, uint32_t *out, size_t cap, int use_cache);

/* Dump the pre-tokenizer pieces, one UTF-8 line per piece and "#ITEM#" between items, for compare.py
 * to check the split against Python's _split. */
int sb_split_dump(sb_tok *t, const char *text, size_t len, FILE *f);

#endif
