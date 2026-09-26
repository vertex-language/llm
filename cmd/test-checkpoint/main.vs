// model.Checkpoint and a family over it, both formats of one model:
// testdata/tiny-llama (hf.co/hf-internal-testing/tiny-random-LlamaForCausalLM,
// a Hugging Face snapshot) and the GGUF llama.WriteGGUF makes of it
// (testdata/tiny-llama.gguf, from `vsc run convert`).
//
// The same weights in both formats must give the same logits: that checks
// the config aliases, the tensor names, the shape order and both RoPE
// layouts at once. And llama.cpp, running the converted file, is the
// oracle for both (golden/tiny-llama.txt, from testdata/oracle/llama_run.cpp).
package main

import (
    "fs"
    "gpu"
    "llm/arch"
    "llm/llama"
    "model"
)

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

let hf = try model.Open(fs.Path("testdata/tiny-llama"))
let gg = try model.Open(fs.Path("testdata/tiny-llama.gguf"))
gg.Aliases = arch.DecoderAliases

// Detection and the config view.
check(hf.Format == .safetensors && hf.Architecture == "LlamaForCausalLM", "a snapshot: safetensors, LlamaForCausalLM")
check(gg.Format == .gguf && gg.Architecture == "llama", "a GGUF: general.architecture llama")
check(llama.Names.contains(hf.Architecture) && llama.Names.contains(gg.Architecture), "llama claims both names")
for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "max_position_embeddings"] {
    check(hf.Config.Int(key) != nil && hf.Config.Int(key) == gg.Config.Int(key), "\(key): \(hf.Config.Int(key) ?? -1) in both")
}
check(hf.Config.Double("rms_norm_eps") == 1e-6 && gg.Config.Double("rms_norm_eps")! - 1e-6 < 1e-12, "rms_norm_eps in both (the GGUF's is f32)")
check(hf.Config.Double("rope_theta") == 10000 && gg.Config.Double("rope_theta") == 10000, "rope_theta in both")
// This test model's config.json says EOS is 1 where its tokenizer.model
// says 2; the GGUF carries the tokenizer's, as llama.cpp's converter does.
check(hf.Config.Ints("eos_token_id") == [1] && gg.Config.Ints("eos_token_id") == [2], "eos_token_id: config.json's, and the GGUF's through the alias to tokenizer.ggml")
check(hf.Config.Raw?["torch_dtype"]?.String == "float32" && gg.Config.Raw == nil, "the raw config.json, where there is one")

// Tensor names and shapes, outermost first in both.
check(arch.GGUFName("model.layers.11.self_attn.q_proj.weight") == "blk.11.attn_q.weight", "GGUFName: a layer's projection")
check(arch.GGUFName("model.embed_tokens.weight") == "token_embd.weight" && arch.GGUFName("lm_head.weight") == "output.weight", "GGUFName: embedding and head")
check(arch.GGUFName("model.layers.x.mlp.up_proj.weight") == nil && arch.GGUFName("model.vision.proj") == nil, "GGUFName: no guess outside the table")
check(hf.Names.count == 21 && gg.Names.count == 21, "21 tensors in each")
for name in hf.Names {
    let a = hf.Info(name)!, b = gg.Info(name)
    check(b != nil && a.Shape == b!.Shape && a.DType == b!.DType, "\(name): \(a.Shape) \(a.SourceType) in both")
}
check(gg.Info("blk.0.attn_q.weight")?.Shape == [16, 16], "a GGUF tensor by its own name")

// Tokenizers: tokenizer.model and the GGUF's keys; and Llama 2's, which
// stories15M's GGUF also carries.
do {
    let a = try hf.Tokenizer(), b = try gg.Tokenizer()
    let stories = try model.Open(fs.Path("testdata/stories15M-q4_0.gguf")).Tokenizer()
    check(a.Vocab.Tokens == b.Vocab.Tokens && a.Vocab.Scores == b.Vocab.Scores && a.Vocab.Kinds == b.Vocab.Kinds,
          "tokenizer.model and the GGUF written from it: the same \(a.Count) tokens")
    // All but one token: llama2.c's export of the vocabulary wrote U+FEFF
    // (id 30143) as an empty string.
    var differ: [int] = []
    for i in 0..<a.Count where a.Vocab.Tokens[i] != stories.Vocab.Tokens[i] || a.Vocab.Scores[i] != stories.Vocab.Scores[i] || a.Vocab.Kinds[i] != stories.Vocab.Kinds[i] {
        differ.append(i)
    }
    check(differ == [30143] && a.Vocab.Tokens[30143] == "\u{FEFF}", "tokenizer.model is Llama 2's vocabulary as stories15M carries it, but for the BOM llama2.c dropped")
    check(a.Vocab.Bos == 1 && a.Vocab.Eos == 2 && a.Vocab.Unknown == 0 && a.Vocab.AddBos && !a.Vocab.AddEos, "special ids and adding BOS")
    check(a.Encode("Once upon a time") == [1, 9038, 2501, 263, 931], "\"Once upon a time\" as llama.cpp tokenizes it")
} catch {
    check(false, "tokenizers: \(error)")
}

// The model through both formats, against each other and llama.cpp.
let golden = try fs.ReadText(fs.Path("cmd/test-checkpoint/golden/tiny-llama.txt")).split(separator: "\n")
var steps: [(int, [(int, float64)])] = []
var greedy: [int] = []
for line in golden {
    let f = line.split(separator: " ")
    if f[0] == "greedy" {
        greedy = f.dropFirst().map { int(String($0))! }
        continue
    }
    var top: [(int, float64)] = []
    for i in 5..<10 {
        let kv = f[i].split(separator: ":")
        top.append((int(String(kv[0]))!, float64(String(kv[1]))!))
    }
    steps.append((int(String(f[3]))!, top))
}
for d in [gpu.CPU(), gpu.Default()] {
    let a = try await llama.Model.Load(hf, on: d)
    let b = try await llama.Model.Load(gg, on: d)
    check(a.Config.HeadDim == 4 && !a.Config.TiedEmbeddings && b.Config.Layers == 2, "\(d.Name): both configs read")
    var between: float64 = 0
    var oracle: float64 = 0
    var same = true
    var position = 0
    for (token, top) in steps {
        let la = try await a.Forward(token, position: position).Download()
        let lb = try await b.Forward(token, position: position).Download()
        position += 1
        for i in 0..<la.count {
            between = max(between, (float64(la[i]) - float64(lb[i])).magnitude)
        }
        for (id, want) in top {
            oracle = max(oracle, (float64(la[id]) - want).magnitude)
        }
        if llama.Argmax(la) != top[0].0 { same = false }
    }
    check(between < 1e-5, "\(d.Name): safetensors and GGUF, the same logits at all \(steps.count) steps (at most \(between) apart)")
    check(oracle < 1e-4, "\(d.Name): llama.cpp's top logits, from the safetensors (at most \(oracle) off)")
    check(same, "\(d.Name): each step's next token is llama.cpp's")
    let a2 = try await llama.Model.Load(hf, on: d)
    let got = try await a2.Generate("Once upon a time", tokens: greedy.count)
    check(got == greedy, "\(d.Name): \(greedy.count) greedy tokens from the safetensors, as llama.cpp's from the GGUF")
    if got != greedy { print("      got  \(got)\n      want \(greedy)") }
}
print(failures == 0 ? "all passed" : "\(failures) failed")
