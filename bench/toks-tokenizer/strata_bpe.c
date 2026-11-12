/* strata_bpe.c - the algorithm of tools/strata_tokenizer.py in C, for the bench/toks-tokenizer
 * experiment.  Parts are vocabulary IDS rather than strings: a merge's result is the token the merge
 * list names, so the final ids are the parts themselves and no string lookup happens in the loop.
 */
#include "strata_bpe.h"
#include "unicode_ranges.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NOID 0xFFFFFFFFu
#define NORANK 0xFFFFFFFFu
#define CACHE_MAX 200000u          /* Tokenizer.PIECE_CACHE_MAX */
#define HEAP_MIN 64                /* Tokenizer.HEAP_MIN */

static uint64_t fnv(const void *p, size_t n)
{
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 10995386743430837637ull; }
    return h;
}

/* Binary search over a sorted [lo,hi] range table. */
static int in_class(const uint32_t *r, uint32_t n, uint32_t cp)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t m = (lo + hi) / 2;
        if (cp < r[2 * m]) { hi = m; }
        else if (cp > r[2 * m + 1]) { lo = m + 1; }
        else { return 1; }
    }
    return 0;
}

static int is_L(uint32_t c) { return in_class(UR_L, UR_L_N, c); }
static int is_M(uint32_t c) { return in_class(UR_M, UR_M_N, c); }
static int is_N(uint32_t c) { return in_class(UR_N, UR_N_N, c); }
static int is_ws(uint32_t c) { return in_class(UR_WS, UR_WS_N, c); }
static int is_LM(uint32_t c) { return is_L(c) || is_M(c); }

/* The GPT-2 byte encoder, exactly as bytes_to_unicode(): printable bytes map to themselves and the
 * other 68 shift into 256+n in byte order. */
static void build_byte_map(uint32_t *b2u)
{
    uint32_t bs[256], cs[256];
    size_t n = 0;
    for (uint32_t b = 0x21; b <= 0x7E; b++) { bs[n] = b; cs[n] = b; n++; }
    for (uint32_t b = 0xA1; b <= 0xAC; b++) { bs[n] = b; cs[n] = b; n++; }
    for (uint32_t b = 0xAE; b <= 0xFF; b++) { bs[n] = b; cs[n] = b; n++; }
    uint32_t k = 0;
    for (uint32_t b = 0; b < 256; b++) {
        int seen = 0;
        for (size_t i = 0; i < n; i++) { if (bs[i] == b) { seen = 1; break; } }
        if (!seen) { bs[n] = b; cs[n] = 256 + k; n++; k++; }
    }
    for (size_t i = 0; i < 256; i++) { b2u[bs[i]] = cs[i]; }
}

/* shortcut: invalid UTF-8 bytes pass through as their own codepoint; the corpus is valid UTF-8, and
 * Python's str.decode would raise instead. Upgrade if the harness ever feeds it raw bytes. */
static size_t utf8_decode(const char *s, size_t len, uint32_t *cp)
{
    size_t i = 0, o = 0;
    while (i < len) {
        uint8_t b = (uint8_t)s[i];
        if (b < 0x80) { cp[o++] = b; i++; }
        else if ((b >> 5) == 6 && i + 1 < len && ((uint8_t)s[i + 1] >> 6) == 2) {
            cp[o++] = ((b & 0x1Fu) << 6) | ((uint8_t)s[i + 1] & 0x3Fu); i += 2;
        } else if ((b >> 4) == 14 && i + 2 < len && ((uint8_t)s[i + 1] >> 6) == 2 && ((uint8_t)s[i + 2] >> 6) == 2) {
            cp[o++] = ((b & 0x0Fu) << 12) | (((uint8_t)s[i + 1] & 0x3Fu) << 6) | ((uint8_t)s[i + 2] & 0x3Fu); i += 3;
        } else if ((b >> 3) == 30 && i + 3 < len && ((uint8_t)s[i + 1] >> 6) == 2 && ((uint8_t)s[i + 2] >> 6) == 2 && ((uint8_t)s[i + 3] >> 6) == 2) {
            cp[o++] = ((b & 0x07u) << 18) | (((uint8_t)s[i + 1] & 0x3Fu) << 12) | (((uint8_t)s[i + 2] & 0x3Fu) << 6) | ((uint8_t)s[i + 3] & 0x3Fu); i += 4;
        } else { cp[o++] = b; i++; }
    }
    return o;
}

static size_t utf8_put(uint32_t c, char *out)
{
    if (c < 0x80) { out[0] = (char)c; return 1; }
    if (c < 0x800) { out[0] = (char)(0xC0 | (c >> 6)); out[1] = (char)(0x80 | (c & 0x3F)); return 2; }
    if (c < 0x10000) { out[0] = (char)(0xE0 | (c >> 12)); out[1] = (char)(0x80 | ((c >> 6) & 0x3F)); out[2] = (char)(0x80 | (c & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (c >> 18)); out[1] = (char)(0x80 | ((c >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((c >> 6) & 0x3F)); out[3] = (char)(0x80 | (c & 0x3F));
    return 4;
}


/* ------------------------------------------------------------------ tables */

typedef struct { uint64_t key; uint32_t pos, left, right; } hnode;

struct sb_tok {
    uint32_t nvocab, nmerges;
    char *tokblob; uint32_t *tok_off, *tok_len;      /* token bytes, by id */
    uint32_t *merge_id;                             /* rank -> id of the merged token */
    uint32_t b2u[256];                              /* byte -> mapped codepoint */
    uint32_t cid[0x300];                            /* mapped codepoint -> vocab id */

    uint32_t sh_mask; uint64_t *sh_key; uint32_t *sh_val;   /* token bytes -> id */
    uint32_t rh_mask; uint64_t *rh_key; uint32_t *rh_val;   /* (left,right) -> rank + 1 */

    uint32_t ch_mask; uint32_t *ch_slot;                    /* piece cache: piece -> entry */
    uint32_t *ent_cps_off, *ent_cps_len, *ent_ids_off, *ent_ids_len;
    uint32_t n_ents;
    uint32_t *cache_cps, *cache_ids; size_t cache_cps_len, cache_ids_len, cache_cps_cap, cache_ids_cap;

    uint32_t *cps, *bytes, *parts;                /* scratch reused per call */
    size_t cps_cap, bytes_cap, parts_cap;
    int32_t *nxt, *prv; hnode *heap; size_t scratch_n;
};

static uint32_t str_id(const sb_tok *t, const char *s, size_t n)
{
    uint64_t h = fnv(s, n);
    uint32_t i = (uint32_t)h & t->sh_mask;
    while (t->sh_val[i] != NOID) {
        if (t->sh_key[i] == h) {
            uint32_t id = t->sh_val[i];
            if (t->tok_len[id] == n && memcmp(t->tokblob + t->tok_off[id], s, n) == 0) { return id; }
        }
        i = (i + 1) & t->sh_mask;
    }
    return NOID;
}

static uint32_t rank_of(const sb_tok *t, uint32_t left, uint32_t right)
{
    uint64_t key = ((uint64_t)left << 32) | right;
    uint32_t i = (uint32_t)key & t->rh_mask;
    while (t->rh_val[i] != 0) {
        if (t->rh_key[i] == key) { return t->rh_val[i] - 1; }
        i = (i + 1) & t->rh_mask;
    }
    return NORANK;
}

static uint32_t pow2_above(uint64_t n) { uint32_t p = 1; while (p < n) { p <<= 1; } return p; }

/* ------------------------------------------------------------------ loading */

/* vocab.json is a JSON object of token -> id written by tools/extract.py with ensure_ascii=False, so
 * the bytes are raw UTF-8 with the usual escapes. */
static int json_string(const char *buf, size_t len, size_t *i, char *tmp, size_t *tmp_len, size_t tmp_cap, char *err, size_t errlen)
{
    if (*i >= len || buf[*i] != '"') { snprintf(err, errlen, "expected a string at offset %zu", *i); return -1; }
    *tmp_len = 0;
    size_t p = *i + 1;
    while (p < len) {
        char c = buf[p];
        if (c == '"') { *i = p + 1; return 0; }
        if (c != '\\') {
            if (*tmp_len + 1 > tmp_cap) { return -2; }
            tmp[(*tmp_len)++] = c;
            p++;
            continue;
        }
        if (p + 1 >= len) { break; }
        char e = buf[p + 1];
        p += 2;
        uint32_t cp = 0;
        int n = 1;
        switch (e) {
            case '"': cp = '"'; break;
            case '\\': cp = '\\'; break;
            case '/': cp = '/'; break;
            case 'b': cp = '\b'; break;
            case 'f': cp = '\f'; break;
            case 'n': cp = '\n'; break;
            case 'r': cp = '\r'; break;
            case 't': cp = '\t'; break;
            case 'u': {
                if (p + 4 > len) { snprintf(err, errlen, "short \\u at %zu", p); return -1; }
                uint32_t v = 0;
                for (int k = 0; k < 4; k++) {
                    char hx = buf[p + k];
                    uint32_t d = (hx <= '9') ? (uint32_t)(hx - '0') : (uint32_t)((hx & 7) + 9);
                    v = (v << 4) | d;
                }
                p += 4;
                cp = v;
                n = 0;
                if (v >= 0xD800 && v <= 0xDBFF && p + 6 <= len && buf[p] == '\\' && buf[p + 1] == 'u') {
                    uint32_t lo = 0;
                    for (int k = 0; k < 4; k++) {
                        char hx = buf[p + 2 + k];
                        uint32_t d = (hx <= '9') ? (uint32_t)(hx - '0') : (uint32_t)((hx & 7) + 9);
                        lo = (lo << 4) | d;
                    }
                    if (lo >= 0xDC00 && lo <= 0xDFFF) { cp = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00); p += 6; }
                }
                if (*tmp_len + 4 > tmp_cap) { return -2; }
                *tmp_len += utf8_put(cp, tmp + *tmp_len);
                continue;
            }
            default: snprintf(err, errlen, "unknown escape \\%c at %zu", e, p - 1); return -1;
        }
        if (*tmp_len + 1 > tmp_cap) { return -2; }
        tmp[(*tmp_len)++] = (char)cp;
        (void)n;
    }
    snprintf(err, errlen, "unterminated string at %zu", p);
    return -1;
}

static int skip_ws(const char *buf, size_t len, size_t *i)
{
    while (*i < len && (buf[*i] == ' ' || buf[*i] == '\n' || buf[*i] == '\r' || buf[*i] == '\t')) { (*i)++; }
    return *i < len ? 0 : -1;
}
static char *read_all(const char *path, size_t *len, char *err, size_t errlen)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) { snprintf(err, errlen, "cannot open %s", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (buf == NULL || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        snprintf(err, errlen, "cannot read %s", path);
        free(buf); fclose(f); return NULL;
    }
    buf[sz] = '\0';
    fclose(f);
    *len = (size_t)sz;
    return buf;
}

int sb_load(sb_tok **out, const char *vocab_json, const char *merges_txt, char *err, size_t errlen)
{
    memset(err, 0, errlen);
    sb_tok *t = (sb_tok *)calloc(1, sizeof(sb_tok));
    for (size_t i = 0; i < 0x300; i++) { t->cid[i] = NOID; }
    build_byte_map(t->b2u);

    size_t vlen = 0;
    char *vbuf = read_all(vocab_json, &vlen, err, errlen);
    if (vbuf == NULL) { sb_free(t); return -1; }
    size_t i = 0;
    if (skip_ws(vbuf, vlen, &i) != 0 || vbuf[i] != '{') {
        snprintf(err, errlen, "vocab.json is not a JSON object"); sb_free(t); free(vbuf); return -1;
    }
    i++;
    size_t tok_cap = 1024, blob_cap = 4096, blob_fill = 0;
    t->tok_off = (uint32_t *)calloc(tok_cap, sizeof(uint32_t));
    t->tok_len = (uint32_t *)calloc(tok_cap, sizeof(uint32_t));
    t->tokblob = (char *)malloc(blob_cap);
    char *tmp = (char *)malloc(4096);
    size_t tmp_len = 0;
    uint32_t max_id = 0;
    for (;;) {
        if (skip_ws(vbuf, vlen, &i) != 0) { snprintf(err, errlen, "vocab.json truncated"); break; }
        if (vbuf[i] == '}') { i++; break; }
        if (json_string(vbuf, vlen, &i, tmp, &tmp_len, 4096, err, errlen) != 0) { break; }
        if (skip_ws(vbuf, vlen, &i) != 0 || vbuf[i] != ':') { snprintf(err, errlen, "vocab.json: expected ':'"); break; }
        i++;
        if (skip_ws(vbuf, vlen, &i) != 0) { snprintf(err, errlen, "vocab.json truncated"); break; }
        uint64_t id = 0;
        if (vbuf[i] < '0' || vbuf[i] > '9') { snprintf(err, errlen, "vocab.json: id is not a number at %zu", i); break; }
        while (i < vlen && vbuf[i] >= '0' && vbuf[i] <= '9') { id = id * 10 + (uint64_t)(vbuf[i++] - '0'); }
        if (id > 0xFFFFFFFEull) { snprintf(err, errlen, "vocab.json: id %llu too large", (unsigned long long)id); break; }
        if (id + 1 > tok_cap) {
            uint32_t newcap = pow2_above(id + 1);
            t->tok_off = (uint32_t *)realloc(t->tok_off, newcap * sizeof(uint32_t));
            t->tok_len = (uint32_t *)realloc(t->tok_len, newcap * sizeof(uint32_t));
            for (uint32_t k = tok_cap; k < newcap; k++) { t->tok_len[k] = 0; }
            tok_cap = newcap;
        }
        if (t->tok_len[(size_t)id] != 0) { snprintf(err, errlen, "vocab.json: id %llu given twice", (unsigned long long)id); break; }
        size_t blob = blob_fill;                 /* running fill position in the blob */
        if (blob + tmp_len + 1 > blob_cap) {
            blob_cap = pow2_above(blob + tmp_len + 1);
            t->tokblob = (char *)realloc(t->tokblob, blob_cap);
        }
        memcpy(t->tokblob + blob, tmp, tmp_len);
        t->tok_off[(size_t)id] = (uint32_t)blob;
        t->tok_len[(size_t)id] = (uint32_t)tmp_len;
        blob_fill = blob + tmp_len;
        if (id > max_id) { max_id = (uint32_t)id; }
        if (skip_ws(vbuf, vlen, &i) != 0) { snprintf(err, errlen, "vocab.json truncated"); break; }
        if (vbuf[i] == ',') { i++; }
    }
    free(vbuf); free(tmp);
    t->nvocab = max_id + 1;
    if (err[0] != '\0') { sb_free(t); return -1; }

    /* token bytes -> id */
    uint32_t sh_cap = pow2_above((uint64_t)t->nvocab * 2);
    t->sh_mask = sh_cap - 1;
    t->sh_key = (uint64_t *)calloc(sh_cap, sizeof(uint64_t));
    t->sh_val = (uint32_t *)calloc(sh_cap, sizeof(uint32_t));
    for (uint32_t s = 0; s < sh_cap; s++) { t->sh_val[s] = NOID; }
    for (uint32_t id = 0; id < t->nvocab; id++) {
        if (t->tok_len[id] == 0) { continue; }
        uint64_t h = fnv(t->tokblob + t->tok_off[id], t->tok_len[id]);
        uint32_t s = (uint32_t)h & t->sh_mask;
        while (t->sh_val[s] != NOID) { s = (s + 1) & t->sh_mask; }
        t->sh_key[s] = h; t->sh_val[s] = id;
    }
    /* every mapped codepoint must be a one-character vocab token, as bytes_to_unicode() assumes */
    char one[4];
    for (uint32_t b = 0; b < 256; b++) {
        uint32_t cp = t->b2u[b];
        size_t n = utf8_put(cp, one);
        uint32_t id = str_id(t, one, n);
        if (id == NOID) { snprintf(err, errlen, "vocab has no byte token for U+%04X", cp); sb_free(t); return -1; }
        t->cid[cp] = id;
    }

    /* merges.txt: rank = line index, exactly as Tokenizer builds self.ranks */
    size_t mlen = 0;
    char *mbuf = read_all(merges_txt, &mlen, err, errlen);
    if (mbuf == NULL) { sb_free(t); return -1; }
    uint32_t nlines = 0;
    for (size_t k = 0; k < mlen; k++) { if (mbuf[k] == '\n') { nlines++; } }
    if (mlen > 0 && mbuf[mlen - 1] != '\n') { nlines++; }
    t->nmerges = nlines;
    t->merge_id = (uint32_t *)calloc(nlines + 1, sizeof(uint32_t));
    uint32_t rh_cap = pow2_above((uint64_t)nlines * 2);
    t->rh_mask = rh_cap - 1;
    t->rh_key = (uint64_t *)calloc(rh_cap, sizeof(uint64_t));
    t->rh_val = (uint32_t *)calloc(rh_cap, sizeof(uint32_t));
    uint32_t rank = 0;
    size_t pos = 0;
    while (pos < mlen) {
        size_t eol = pos;
        while (eol < mlen && mbuf[eol] != '\n') { eol++; }
        if (eol > pos) {
            size_t sp = pos;
            while (sp < eol && mbuf[sp] != ' ') { sp++; }
            if (sp >= eol || memchr(mbuf + sp + 1, ' ', eol - sp - 1) != NULL) {
                snprintf(err, errlen, "merge %u is not a pair", rank); break;
            }
            uint32_t left = str_id(t, mbuf + pos, sp - pos);
            uint32_t right = str_id(t, mbuf + sp + 1, eol - sp - 1);
            if (left == NOID || right == NOID) { snprintf(err, errlen, "merge %u names a token outside the vocab", rank); break; }
            uint32_t l = t->tok_len[left], r = t->tok_len[right];
            char *joined = (char *)malloc(l + r);
            memcpy(joined, t->tokblob + t->tok_off[left], l);
            memcpy(joined + l, t->tokblob + t->tok_off[right], r);
            uint32_t merged = str_id(t, joined, l + r);
            free(joined);
            if (merged == NOID) { snprintf(err, errlen, "merge %u names a token outside the vocab", rank); break; }
            t->merge_id[rank] = merged;
            uint64_t key = ((uint64_t)left << 32) | right;
            uint32_t s = (uint32_t)key & t->rh_mask;
            while (t->rh_val[s] != 0) { s = (s + 1) & t->rh_mask; }
            t->rh_key[s] = key; t->rh_val[s] = rank + 1;
        }
        rank++;
        pos = eol + 1;
    }
    free(mbuf);
    if (err[0] != '\0') { sb_free(t); return -1; }

    uint32_t ch_cap = pow2_above((uint64_t)CACHE_MAX * 2);
    t->ch_mask = ch_cap - 1;
    t->ch_slot = (uint32_t *)calloc(ch_cap, sizeof(uint32_t));
    for (uint32_t s = 0; s < ch_cap; s++) { t->ch_slot[s] = NOID; }
    t->ent_cps_off = (uint32_t *)calloc(CACHE_MAX, sizeof(uint32_t));
    t->ent_cps_len = (uint32_t *)calloc(CACHE_MAX, sizeof(uint32_t));
    t->ent_ids_off = (uint32_t *)calloc(CACHE_MAX, sizeof(uint32_t));
    t->ent_ids_len = (uint32_t *)calloc(CACHE_MAX, sizeof(uint32_t));
    *out = t;
    return 0;
}

void sb_free(sb_tok *t)
{
    if (t == NULL) { return; }
    free(t->tokblob); free(t->tok_off); free(t->tok_len); free(t->merge_id);
    free(t->sh_key); free(t->sh_val); free(t->rh_key); free(t->rh_val);
    free(t->ch_slot); free(t->ent_cps_off); free(t->ent_cps_len);
    free(t->ent_ids_off); free(t->ent_ids_len);
    free(t->cache_cps); free(t->cache_ids);
    free(t->cps); free(t->bytes); free(t->parts); free(t->nxt); free(t->prv); free(t->heap);
    free(t);
}


/* ------------------------------------------------------------------ pre-tokenizer */

/* The qwen35 pattern's seven alternatives, tried in order at position p.  Returns the end of the
 * piece, or p when nothing matches (regex findall then skips one character). */
static size_t match_piece(const uint32_t *cps, size_t n, size_t p)
{
    uint32_t c = cps[p];
    /* '[sS]|'[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD] : first characters are disjoint */
    if (c == '\'' && p + 1 < n) {
        uint32_t a = cps[p + 1];
        if (a == 's' || a == 'S' || a == 't' || a == 'T' || a == 'm' || a == 'M' || a == 'd' || a == 'D') { return p + 2; }
        if ((a == 'r' || a == 'R' || a == 'v' || a == 'V') && p + 2 < n && (cps[p + 2] == 'e' || cps[p + 2] == 'E')) { return p + 3; }
        if ((a == 'l' || a == 'L') && p + 2 < n && (cps[p + 2] == 'l' || cps[p + 2] == 'L')) { return p + 3; }
    }
    /* (?[^\r\n\p{L}\p{N})?[\p{L}\p{M}]+ */
    if (is_LM(c)) { size_t e = p; while (e < n && is_LM(cps[e])) { e++; } return e; }
    if (c != '\r' && c != '\n' && !is_L(c) && !is_N(c) && p + 1 < n && is_LM(cps[p + 1])) {
        size_t e = p + 1; while (e < n && is_LM(cps[e])) { e++; } return e;
    }
    /* \p{N} */
    if (is_N(c)) { return p + 1; }
    /* ?[^s\p{L}\p{M}\p{N]+[\r\n]* */
    size_t s = (c == ' ') ? p + 1 : p;
    if (s < n && !is_ws(cps[s]) && !is_LM(cps[s]) && !is_N(cps[s])) {
        size_t e = s;
        while (e < n && !is_ws(cps[e]) && !is_LM(cps[e]) && !is_N(cps[e])) { e++; }
        while (e < n && (cps[e] == '\r' || cps[e] == '\n')) { e++; }
        return e;
    }
    /* \s*[\r\n]+ | \s*(?!\S) | \s+ */
    if (is_ws(c)) {
        size_t e = p;
        int has_nl = 0;
        while (e < n && is_ws(cps[e])) { if (cps[e] == '\r' || cps[e] == '\n') { has_nl = 1; } e++; }
        if (has_nl) { return e; }
        if (e == n || is_ws(cps[e])) { return e; }
        if (e - p >= 2) { return e - 1; }
        return e;
    }
    return p;
}

/* ------------------------------------------------------------------ BPE */

static size_t bpe_scan(sb_tok *t, uint32_t *parts, size_t n)
{
    while (n > 1) {
        uint32_t best = 0, best_rank = 0;
        int found = 0;
        for (size_t i = 0; i + 1 < n; i++) {
            uint32_t r = rank_of(t, parts[i], parts[i + 1]);
            if (r != NORANK && (!found || r < best_rank)) { best = (uint32_t)i; best_rank = r; found = 1; }
        }
        if (!found) { break; }
        parts[best] = t->merge_id[best_rank];
        memmove(parts + best + 1, parts + best + 2, (n - best - 2) * sizeof(uint32_t));
        n--;
    }
    return n;
}

/* rank is unique per pair, so a heap node never ties with another on more than (rank, pos). */
static int hless(hnode a, hnode b)
{
    if (a.key != b.key) { return a.key < b.key; }
    if (a.pos != b.pos) { return a.pos < b.pos; }
    if (a.left != b.left) { return a.left < b.left; }
    return a.right < b.right;
}

static void hswap(hnode *a, hnode *b) { hnode c = *a; *a = *b; *b = c; }

static void hsift_down(hnode *h, size_t n, size_t i)
{
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && hless(h[l], h[m])) { m = l; }
        if (r < n && hless(h[r], h[m])) { m = r; }
        if (m == i) { break; }
        hswap(&h[i], &h[m]);
        i = m;
    }
}

static size_t bpe_heap(sb_tok *t, uint32_t *parts, size_t n, hnode *h)
{
    int32_t *nxt = t->nxt, *prv = t->prv;
    for (size_t i = 0; i < n; i++) { nxt[i] = (int32_t)(i + 1 < n ? (int32_t)(i + 1) : -1); prv[i] = (i > 0 ? (int32_t)(i - 1) : -1); }
    size_t hn = 0;
    for (size_t i = 0; i + 1 < n; i++) {
        uint32_t r = rank_of(t, parts[i], parts[i + 1]);
        if (r == NORANK) { continue; }
        h[hn].key = r; h[hn].pos = (uint32_t)i; h[hn].left = parts[i]; h[hn].right = parts[i + 1];
        hn++;
    }
    for (size_t i = hn / 2; i-- > 0;) { hsift_down(h, hn, i); }
    while (hn > 0) {
        hnode top = h[0];
        h[0] = h[hn - 1];
        hn--;
        if (hn > 0) { hsift_down(h, hn, 0); }
        size_t i = top.pos;
        int32_t j = nxt[i];
        if (parts[i] != top.left || j < 0 || parts[j] != top.right) { continue; }
        parts[i] = t->merge_id[(uint32_t)top.key];
        parts[j] = NOID;
        int32_t k = nxt[j];
        nxt[i] = k;
        if (k >= 0) {
            prv[k] = i;
            uint32_t r = rank_of(t, parts[i], parts[k]);
            if (r != NORANK) {
                h[hn].key = r; h[hn].pos = (uint32_t)i; h[hn].left = parts[i]; h[hn].right = parts[k];
                hn++;
                size_t u = hn - 1;
                while (u > 0 && hless(h[u], h[(u - 1) / 2])) { hswap(&h[u], &h[(u - 1) / 2]); u = (u - 1) / 2; }
            }
        }
        int32_t p = prv[i];
        if (p >= 0) {
            uint32_t r = rank_of(t, parts[p], parts[i]);
            if (r != NORANK) {
                h[hn].key = r; h[hn].pos = (uint32_t)p; h[hn].left = parts[p]; h[hn].right = parts[i];
                hn++;
                size_t u = hn - 1;
                while (u > 0 && hless(h[u], h[(u - 1) / 2])) { hswap(&h[u], &h[(u - 1) / 2]); u = (u - 1) / 2; }
            }
        }
    }
    size_t out = 0;
    for (size_t i = 0; i < n; i++) { if (parts[i] != NOID) { parts[out++] = parts[i]; } }
    return out;
}


/* ------------------------------------------------------------------ piece cache */

/* Keyed by the piece's codepoints; mirrors Tokenizer._piece_ids and its PIECE_CACHE_MAX cap. */
static void cache_clear(sb_tok *t)
{
    for (uint32_t s = 0; s <= t->ch_mask; s++) { t->ch_slot[s] = NOID; }
    t->n_ents = 0;
    t->cache_cps_len = 0;
    t->cache_ids_len = 0;
}

static int cache_find(sb_tok *t, const uint32_t *cps, size_t n, uint32_t *ent)
{
    uint64_t h = fnv(cps, n * sizeof(uint32_t));
    uint32_t s = (uint32_t)h & t->ch_mask;
    while (t->ch_slot[s] != NOID) {
        uint32_t e = t->ch_slot[s];
        if (t->ent_cps_len[e] == n && memcmp(t->cache_cps + t->ent_cps_off[e], cps, n * sizeof(uint32_t)) == 0) {
            *ent = e; return 1;
        }
        s = (s + 1) & t->ch_mask;
    }
    return 0;
}

static void cache_store(sb_tok *t, const uint32_t *cps, size_t n, const uint32_t *ids, size_t ni)
{
    if (t->n_ents >= CACHE_MAX) { return; }
    if (t->cache_cps_len + n > t->cache_cps_cap) {
        t->cache_cps_cap = pow2_above((t->cache_cps_len + n) * 2);
        t->cache_cps = (uint32_t *)realloc(t->cache_cps, t->cache_cps_cap * sizeof(uint32_t));
    }
    if (t->cache_ids_len + ni > t->cache_ids_cap) {
        t->cache_ids_cap = pow2_above((t->cache_ids_len + ni) * 2);
        t->cache_ids = (uint32_t *)realloc(t->cache_ids, t->cache_ids_cap * sizeof(uint32_t));
    }
    uint32_t e = t->n_ents++;
    t->ent_cps_off[e] = (uint32_t)t->cache_cps_len;
    t->ent_cps_len[e] = (uint32_t)n;
    memcpy(t->cache_cps + t->cache_cps_len, cps, n * sizeof(uint32_t));
    t->cache_cps_len += n;
    t->ent_ids_off[e] = (uint32_t)t->cache_ids_len;
    t->ent_ids_len[e] = (uint32_t)ni;
    memcpy(t->cache_ids + t->cache_ids_len, ids, ni * sizeof(uint32_t));
    t->cache_ids_len += ni;
    uint64_t h = fnv(cps, n * sizeof(uint32_t));
    uint32_t s = (uint32_t)h & t->ch_mask;
    while (t->ch_slot[s] != NOID) { s = (s + 1) & t->ch_mask; }
    t->ch_slot[s] = e;
}

/* ------------------------------------------------------------------ encode */

static void ensure(sb_tok *t, size_t n)
{
    if (n <= t->scratch_n) { return; }
    t->cps = (uint32_t *)realloc(t->cps, n * sizeof(uint32_t));
    t->parts = (uint32_t *)realloc(t->parts, n * sizeof(uint32_t));
    t->nxt = (int32_t *)realloc(t->nxt, n * sizeof(int32_t));
    t->prv = (int32_t *)realloc(t->prv, n * sizeof(int32_t));
    t->heap = (hnode *)realloc(t->heap, (3 * n + 8) * sizeof(hnode));
    t->scratch_n = n;
}

/* piece codepoints -> utf-8 bytes -> byte-level codepoints -> vocab ids */
static size_t piece_to_ids(sb_tok *t, const uint32_t *cps, size_t n, uint32_t *ids)
{
    size_t ni = 0;
    for (size_t i = 0; i < n; i++) {
        char buf[4];
        size_t k = utf8_put(cps[i], buf);
        for (size_t b = 0; b < k; b++) { ids[ni++] = t->cid[t->b2u[(uint8_t)buf[b]]]; }
    }
    return ni;
}

int sb_encode(sb_tok *t, const char *text, size_t len, uint32_t *out, size_t cap, int use_cache)
{
    if (!use_cache) { cache_clear(t); }
    ensure(t, 4 * len + 8);
    size_t ncp = utf8_decode(text, len, t->cps);
    size_t total = 0;
    size_t p = 0;
    while (p < ncp) {
        size_t e = match_piece(t->cps, ncp, p);
        if (e == p) { p++; continue; }   /* nothing matched: findall skips the character */
        size_t plen = e - p;
        uint32_t ent = 0;
        if (use_cache && cache_find(t, t->cps + p, plen, &ent)) {
            size_t ni = t->ent_ids_len[ent];
            if (total + ni > cap) { return -1; }
            memcpy(out + total, t->cache_ids + t->ent_ids_off[ent], ni * sizeof(uint32_t));
            total += ni;
        } else {
            size_t ni = piece_to_ids(t, t->cps + p, plen, t->parts);
            ni = (ni > HEAP_MIN) ? bpe_heap(t, t->parts, ni, t->heap) : bpe_scan(t, t->parts, ni);
            if (total + ni > cap) { return -1; }
            memcpy(out + total, t->parts, ni * sizeof(uint32_t));
            total += ni;
            if (use_cache) { cache_store(t, t->cps + p, plen, t->parts, ni); }
        }
        p = e;
    }
    return (int)total;
}

int sb_split_dump(sb_tok *t, const char *text, size_t len, FILE *f)
{
    ensure(t, len + 8);
    size_t ncp = utf8_decode(text, len, t->cps);
    size_t p = 0;
    while (p < ncp) {
        size_t e = match_piece(t->cps, ncp, p);
        if (e == p) { p++; continue; }
        for (size_t i = p; i < e; i++) {
            char buf[4];
            size_t k = utf8_put(t->cps[i], buf);
            fwrite(buf, 1, k, f);
        }
        fputc('\n', f);
        p = e;
    }
    fputs("#ITEM#\n", f);
    return 0;
}
