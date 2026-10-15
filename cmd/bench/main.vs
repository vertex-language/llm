// Speed: a model's load time; the tokens a second it reads a prompt of
// PROMPT tokens in one batch (llama-bench's pp); and the tokens a second
// it generates greedily after a short prompt (llama-bench's tg), on each
// device.
//
//   MODEL=testdata/stories110M-q4_k_m.gguf PROMPT=512 TOKENS=100 vsc run bench
package main

import (
    "fs"
    "gpu"
    "llm/arch"
    "llm/llama"
    "os/env"
    "time"
)

let path = env.Get("MODEL") ?? "testdata/stories15M-q4_0.gguf"
let n = int(env.Get("TOKENS") ?? "100") ?? 100
let prompt = int(env.Get("PROMPT") ?? "128") ?? 128
for d in [gpu.Default(), gpu.CPU()] {
    var t = time.Instant.Now()
    let m = try await llama.Model.Load(fs.Path(path), on: d)
    let load = t.Elapsed()
    // A prompt of synthetic tokens, once to warm up and once timed.
    let ids = (0..<min(prompt, m.Context)).map { ($0 * 31 + 7) % m.Network.Vocab }
    _ = try await m.Decode(arch.Batch.Prompt(ids), arch.Output.logits).Download()
    t = time.Instant.Now()
    _ = try await m.Decode(arch.Batch.Prompt(ids), arch.Output.logits).Download()
    let pp = t.Elapsed()
    print("\(d.Name): prompt of \(ids.count) tokens in \(pp): \(int(float64(ids.count) / pp.AsSeconds())) tokens/s")
    t = time.Instant.Now()
    let out = try await m.Generate("Once upon a time", tokens: n)
    let gen = t.Elapsed()
    let perSecond = float64(out.count + 5) / gen.AsSeconds()
    print("\(d.Name): load \(load), \(out.count) tokens in \(gen): \(int(perSecond)) tokens/s")
}
