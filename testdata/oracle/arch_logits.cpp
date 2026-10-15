// The oracle for every family: llama.cpp (libllama, CPU only) given one of
// test-llama-archs's random-weight models and a fixed run of tokens as a
// single batch, written as the logits of every position. llm's
// `test-archs` runs the same tokens through the Vertex family -- as one
// batch, and again split between a batch and single tokens -- and
// compares (gen-archs.sh makes the models and runs this on each).
//
//   c++ -std=c++17 -I$LLAMA/include -I$LLAMA/ggml/include arch_logits.cpp -L$BUILD/bin -lllama -o arch_logits
//   ./arch_logits model.gguf 48 > model.logits
//
// The file: "VXLG", then int32 n_tokens and n_vocab, the tokens as int32,
// then n_tokens rows of n_vocab float32 logits; little-endian.
#include "llama.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: arch_logits model.gguf n_tokens\n"); return 2; }
    llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int32_t n = atoi(argv[2]);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = n; cp.n_ubatch = n; cp.n_threads = 4; cp.no_perf = true;
    // The KV cache in float32, as Vertex keeps it: llama.cpp's f16 default
    // rounds every key and value.
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot make a context for %s\n", argv[1]); return 1; }

    // A fixed run of tokens: every one in the vocabulary is reachable, and
    // none repeats in the first n_vocab / gcd steps.
    std::vector<llama_token> tokens(n);
    for (int32_t i = 0; i < n; i++) tokens[i] = (i * 31 + 7) % n_vocab;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int32_t i = 0; i < n; i++) {
        b.token[i] = tokens[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = n;
    if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed for %s\n", argv[1]); return 1; }

    fwrite("VXLG", 1, 4, stdout);
    fwrite(&n, 4, 1, stdout);
    fwrite(&n_vocab, 4, 1, stdout);
    fwrite(tokens.data(), 4, n, stdout);
    for (int32_t i = 0; i < n; i++) fwrite(llama_get_logits_ith(ctx, i), 4, n_vocab, stdout);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
