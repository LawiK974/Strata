// include/strata/tokenizer/toks_tokenizer.hpp - a pack tokenizer driven by toks
// (https://github.com/actual-computer/toks).
//
// A Strata pack ships its own format: vocab.json (token -> id), merges.txt (rank = line order),
// token_type.json (one type per id) and tokenizer.json (model, pre, pre_pattern, vocab_size, n_merges,
// special_ids).  toks reads HuggingFace tokenizer.json, so this reads the pack's files and builds that
// document in memory: the same vocab, the same merges in the same rank order, the pack's own pre_pattern,
// no normalizer (Strata does not NFC-normalize) and a ByteLevel decoder.  It never writes the pack's files.
//
// Header-only, like strata_artifact: the state is one toks context plus its scratch.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "toks.h"

namespace strata {

// --- the JSON reader the pack's four files need: object, array, string, integer, null -------------
// Handwritten because the repo has no JSON parser in C++ and none is added for this.  The pack's files are
// flat (one object of strings, one array of ints, one small object), so a bounded recursive reader is
// enough; a file nested deeper than 32 levels is refused rather than recursed.
struct Json {
    enum Kind { Null, Int, String, Array, Object } kind = Null;
    long long num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json *find(const char *key) const {
        for (const auto &kv : obj) {
            if (kv.first == key) { return &kv.second; }
        }
        return nullptr;
    }
};

class JsonReader {
  public:
    explicit JsonReader(const std::string &text) : t_(text) {}

    bool value(Json &out, std::string &err) { return value(out, err, 0); }

  private:
    const std::string &t_;
    size_t i_ = 0;

    void skip_ws() {
        while (i_ < t_.size()) {
            const char c = t_[i_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') { break; }
            i_++;
        }
    }

    bool value(Json &out, std::string &err, int depth) {
        if (depth > 32) { err = "JSON nested deeper than 32 levels"; return false; }
        skip_ws();
        if (i_ >= t_.size()) { err = "JSON ended early"; return false; }
        const char c = t_[i_];
        if (c == '{') { return object(out, err, depth); }
        if (c == '[') { return array(out, err, depth); }
        if (c == '"' || c == '-' || (c >= '0' && c <= '9')) { return scalar(out, err); }
        if (t_.compare(i_, 4, "null") == 0) { i_ += 4; out.kind = Json::Null; return true; }
        if (t_.compare(i_, 4, "true") == 0) { i_ += 4; out.kind = Json::Int; out.num = 1; return true; }
        if (t_.compare(i_, 5, "false") == 0) { i_ += 5; out.kind = Json::Int; out.num = 0; return true; }
        err = "JSON has a value that is not an object, array, string or number";
        return false;
    }

    bool object(Json &out, std::string &err, int depth) {
        out.kind = Json::Object;
        i_++;  // '{'
        skip_ws();
        if (i_ < t_.size() && t_[i_] == '}') { i_++; return true; }
        for (;;) {
            skip_ws();
            Json key;
            if (!scalar(key, err) || key.kind != Json::String) {
                err = "JSON object key is not a string at byte " + std::to_string(i_);
                return false;
            }
            skip_ws();
            if (i_ >= t_.size() || t_[i_] != ':') { err = "JSON object key has no ':'"; return false; }
            i_++;
            Json val;
            if (!value(val, err, depth + 1)) { return false; }
            out.obj.emplace_back(key.str, std::move(val));
            skip_ws();
            if (i_ < t_.size() && t_[i_] == ',') { i_++; continue; }
            if (i_ < t_.size() && t_[i_] == '}') { i_++; return true; }
            err = "JSON object is not closed";
            return false;
        }
    }

    bool array(Json &out, std::string &err, int depth) {
        out.kind = Json::Array;
        i_++;  // '['
        skip_ws();
        if (i_ < t_.size() && t_[i_] == ']') { i_++; return true; }
        for (;;) {
            Json val;
            if (!value(val, err, depth + 1)) { return false; }
            out.arr.push_back(std::move(val));
            skip_ws();
            if (i_ < t_.size() && t_[i_] == ',') { i_++; continue; }
            if (i_ < t_.size() && t_[i_] == ']') { i_++; return true; }
            err = "JSON array is not closed";
            return false;
        }
    }

    bool scalar(Json &out, std::string &err) {
        if (i_ >= t_.size()) { err = "JSON ended early"; return false; }
        if (t_[i_] == '"') {
            out.kind = Json::String;
            i_++;
            while (i_ < t_.size()) {
                const char c = t_[i_];
                if (c == '"') { i_++; return true; }
                if (c != '\\') { out.str.push_back(c); i_++; continue; }
                i_++;
                if (i_ >= t_.size()) { break; }
                const char e = t_[i_];
                if (e == 'n') { out.str.push_back('\n'); }
                else if (e == 't') { out.str.push_back('\t'); }
                else if (e == 'r') { out.str.push_back('\r'); }
                else if (e == 'u') {
                    // \uXXXX written back as UTF-8; a pack's strings are UTF-8 already, so this only
                    // matters for an escaped control byte.
                    if (i_ + 4 >= t_.size()) { err = "JSON escape is cut short"; return false; }
                    unsigned cp = 0;
                    for (int k = 1; k <= 4; k++) {
                        const char h = t_[i_ + k];
                        unsigned d = ('0' <= h && h <= '9') ? unsigned(h - '0') : unsigned((h | 0x20) - ('a' - 10));
                        if (d > 15) { err = "JSON escape is not hex"; return false; }
                        cp = cp * 16 + d;
                    }
                    i_ += 4;
                    if (cp < 0x80) { out.str.push_back(char(cp)); }
                    else if (cp < 0x800) {
                        out.str.push_back(char(0xC0 | (cp >> 6)));
                        out.str.push_back(char(0x80 | (cp & 0x3F)));
                    } else {
                        out.str.push_back(char(0xE0 | (cp >> 12)));
                        out.str.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
                        out.str.push_back(char(0x80 | (cp & 0x3F)));
                    }
                } else { out.str.push_back(e); }
                i_++;
            }
            err = "JSON string is not closed";
            return false;
        }
        out.kind = Json::Int;
        size_t start = i_;
        if (t_[i_] == '-') { i_++; }
        while (i_ < t_.size() && t_[i_] >= '0' && t_[i_] <= '9') { i_++; }
        if (i_ == start) { err = "JSON has no number where one was expected"; return false; }
        out.num = std::stoll(t_.substr(start, i_ - start));
        return true;
    }
};

inline bool read_file(const std::string &path, std::string &out, std::string &err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Escape a string for the JSON document handed to toks: UTF-8 bytes pass through, '"' and '\' are
// escaped, and a byte below 0x20 becomes \u00XX so a control byte cannot break the document.
inline void json_escape(std::string &out, const std::string &s) {
    static const char *HEX = "0123456789abcdef";
    out.push_back('"');
    for (const unsigned char c : s) {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(char(c)); }
        else if (c < 0x20) {
            out += "\\u00";
            out.push_back(HEX[c >> 4]);
            out.push_back(HEX[c & 0xF]);
        } else { out.push_back(char(c)); }
    }
    out.push_back('"');
}

class ToksPackTokenizer {
  public:
    // The pack's own metadata, kept so a caller can use the types and specials without re-reading files.
    std::string model;
    std::string pre;
    std::string pre_pattern;
    uint64_t vocab_size = 0;
    uint64_t n_merges = 0;
    std::vector<uint32_t> types;
    std::vector<std::pair<std::string, uint32_t>> special_ids;

    int load(const std::string &pack_dir, std::string &err) {
        std::string vocab_text, merges_text, types_text, meta_text;
        if (!read_file(pack_dir + "/vocab.json", vocab_text, err)) { return -1; }
        if (!read_file(pack_dir + "/merges.txt", merges_text, err)) { return -1; }
        if (!read_file(pack_dir + "/tokenizer.json", meta_text, err)) { return -1; }

        Json vocab;
        JsonReader vr(vocab_text);
        if (!vr.value(vocab, err) || vocab.kind != Json::Object) { return -1; }
        Json meta;
        JsonReader mr(meta_text);
        if (!mr.value(meta, err) || meta.kind != Json::Object) { return -1; }
        const Json *pattern = meta.find("pre_pattern");
        if (pattern == nullptr || pattern->kind != Json::String || pattern->str.empty()) {
            err = "tokenizer.json has no pre_pattern";
            return -1;
        }
        pre_pattern = pattern->str;
        if (const Json *m = meta.find("model")) { model = m->str; }
        if (const Json *p = meta.find("pre")) { pre = p->str; }
        if (const Json *v = meta.find("vocab_size")) { vocab_size = uint64_t(v->num); }
        if (const Json *n = meta.find("n_merges")) { n_merges = uint64_t(n->num); }
        if (const Json *s = meta.find("special_ids")) {
            for (const auto &kv : s->obj) { special_ids.emplace_back(kv.first, uint32_t(kv.second.num)); }
        }
        if (read_file(pack_dir + "/token_type.json", types_text, err)) {
            Json types_json;
            JsonReader tr(types_text);
            if (!tr.value(types_json, err) || types_json.kind != Json::Array) { return -1; }
            for (const Json &v : types_json.arr) { types.push_back(uint32_t(v.num)); }
        } else if (err.rfind("cannot open", 0) != 0) {
            return -1;
        } else {
            err.clear();  // a pack with no token_type.json is readable; types stay empty
        }

        // The HuggingFace document toks reads, built here from the pack's files.
        std::string doc;
        doc.reserve(vocab_text.size() + merges_text.size() + 512);
        doc += "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"normalizer\":null,";
        doc += "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
               "{\"type\":\"Split\",\"pattern\":{\"Regex\":";
        json_escape(doc, pre_pattern);
        doc += "},\"behavior\":\"Isolated\",\"invert\":false},"
               "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}]},";
        doc += "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\"},";
        doc += "\"model\":{\"type\":\"BPE\",\"vocab\":{";
        for (size_t k = 0; k < vocab.obj.size(); k++) {
            if (k != 0) { doc.push_back(','); }
            json_escape(doc, vocab.obj[k].first);
            doc += ":";
            doc += std::to_string(vocab.obj[k].second.num);
        }
        doc += "},\"merges\":[";
        size_t rank = 0;
        for (size_t pos = 0; pos < merges_text.size();) {
            size_t end = merges_text.find('\n', pos);
            if (end == std::string::npos) { end = merges_text.size(); }
            if (end > pos) {
                if (rank != 0) { doc.push_back(','); }
                json_escape(doc, merges_text.substr(pos, end - pos));
                rank++;
            }
            pos = end + 1;
        }
        doc += "]}}";

        if (n_merges != 0 && rank != n_merges) {
            err = "merges.txt holds " + std::to_string(rank) + " merges, tokenizer.json says " + std::to_string(n_merges);
            return -2;
        }
        n_merges = rank;

        toks_diag diag;
        memset(&diag, 0, sizeof(diag));
        toks_load_opts opts;
        memset(&opts, 0, sizeof(opts));
        opts.size = sizeof(opts);
        opts.diag = &diag;
        const int64_t rc = toks_load_mem_copy(&ctx_, doc.data(), uint64_t(doc.size()), &opts);
        if (rc != 0) {
            err = "toks_load_mem_copy " + std::to_string(rc) + ": " + diag.what;
            return -3;
        }
        toks_info info;
        memset(&info, 0, sizeof(info));
        info.size = sizeof(info);
        toks_get_info(ctx_, &info);
        n_ids = info.n_ids;
        if (vocab_size != 0 && vocab_size != n_ids) {
            err = "toks built " + std::to_string(n_ids) + " ids, the pack says " + std::to_string(vocab_size);
            return -4;
        }
        vocab_size = n_ids;
        return 0;
    }



    void unload() {
        toks_unload(ctx_);
        ctx_ = nullptr;
        scratch_.clear();
    }

    // Plain text with no added token and no post-processing: Tokenizer.encode(text) with
    // parse_special=False.  Returns the number of ids, or a negative toks code.
    int64_t encode(const std::string &text, std::vector<uint32_t> &ids) const {
        if (ctx_ == nullptr) { return TOKS_E_ARG; }
        const uint64_t need = toks_scratch_bytes(ctx_, uint64_t(text.size()), 0);
        if (need > scratch_.size()) {
            scratch_.resize(size_t(need));
            if (toks_scratch_init(ctx_, scratch_.data(), need, 0) < 0) { return TOKS_E_SCRATCH; }
        }
        const uint64_t bound = toks_encode_bound(ctx_, uint64_t(text.size()));
        ids.resize(size_t(bound));
        return toks_encode(ctx_, text.data(), uint64_t(text.size()), TOKS_ADDED_NONE | TOKS_NO_POSTPROCESS,
                           ids.data(), uint32_t(ids.size()), scratch_.data());
    }

    // The bytes the ids spell, never repaired: ByteLevel decoding.  Two passes, the first with cap 0
    // sizes the output.
    int64_t decode(const std::vector<uint32_t> &ids, std::string &out) const {
        if (ctx_ == nullptr) { return TOKS_E_ARG; }
        const int64_t total = toks_decode(ctx_, ids.data(), uint64_t(ids.size()), TOKS_DECODE_RAW, nullptr, 0);
        if (total < 0) { return total; }
        out.resize(size_t(total));
        const int64_t written = toks_decode(ctx_, ids.data(), uint64_t(ids.size()), TOKS_DECODE_RAW,
                                            reinterpret_cast<uint8_t *>(out.data()), uint64_t(total));
        if (written < 0) { return written; }
        out.resize(size_t(written));
        return written;
    }

    uint64_t ids_count() const { return n_ids; }

    // shortcut: token_type.json's types and special_ids are kept but not handed to toks (no HF
    // added_tokens in the document built here); upgrade when the chat template and type ids go
    // through this context.
    bool is_special(uint32_t id) const {
        if (!types.empty() && size_t(id) < types.size()) { return types[size_t(id)] >= 3; }
        for (const auto &s : special_ids) { if (s.second == id) { return true; } }
        return false;
    }

    ~ToksPackTokenizer() { unload(); }

    ToksPackTokenizer() = default;
    ToksPackTokenizer(const ToksPackTokenizer &) = delete;
    ToksPackTokenizer &operator=(const ToksPackTokenizer &) = delete;

  private:
    toks_ctx *ctx_ = nullptr;
    uint64_t n_ids = 0;
    mutable std::vector<uint8_t> scratch_;
};

}  // namespace strata
