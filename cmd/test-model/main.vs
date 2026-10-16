// The facade: references (registry names and local paths, never Hub
// URLs), finding a family, loading local references, and refusing what no
// family claims (llm).
package main

import (
    "fs"
    "gpu"
    "llm"
    "model"
)

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

// References that are refused before anything is fetched.
for (ref, want) in [("hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "Hub references are not loaded directly"),
                    ("https://huggingface.co/Qwen/Qwen3-8B", "Hub references are not loaded directly"),
                    ("no-such-model:7b", "no model is named no-such-model"),
                    ("kokoro", "kokoro:82m is a tts model, not text generation")] {
    do {
        _ = try await llm.Locate(ref)
        check(false, "\(ref) is refused")
    } catch {
        check("\(error)".contains(want), "\(ref) is refused: \(error)")
    }
}

// Families.
let hfCk = try model.Open(fs.Path("testdata/tiny-llama"))
let ggCk = try model.Open(fs.Path("testdata/stories260K.gguf"))
check(llm.FamilyOf(hfCk)?.Name == "llama" && llm.FamilyOf(ggCk)?.Name == "llama", "llama claims both names")

// Loading local references, through the family registry.
for ref in ["testdata/tiny-llama", "testdata/tiny-llama.gguf", "./testdata/stories260K.gguf"] {
    do {
        let m = try await llm.Load(ref, on: gpu.CPU())
        let out = try await llm.Generate(m, "Once upon a time", tokens: 8)
        check(out.count == 8 || out.last == m.Tokenizer.Vocab.Eos, "\(ref): loaded and generated \(out.count) tokens")
    } catch {
        check(false, "\(ref): \(error)")
    }
}
let info = try await llm.Inspect("testdata/tiny-llama")
check(info.Architecture == "LlamaForCausalLM" && info.Family == "llama" && info.Format == .safetensors && info.Tensors == 21 && info.Bytes == 4129088,
      "Inspect: architecture, family, format, tensors, bytes")

// An architecture no family claims: config.json says so, and the error
// lists what is known.
let dir = try fs.TempDir(prefix: "model_")
try fs.WriteText(dir / "config.json", "{\"architectures\": [\"Qwen4ForCausalLM\"]}")
try fs.Copy(fs.Path("testdata/tiny-llama/model.safetensors"), dir / "model.safetensors")
do {
    _ = try await llm.Load(dir.String(), on: gpu.CPU())
    check(false, "an unknown family is refused")
} catch {
    check("\(error)".contains("Qwen4ForCausalLM is not a known family; known: LlamaForCausalLM, llama"), "an unknown family is refused: \(error)")
}
try? fs.RemoveAll(dir)
print(failures == 0 ? "all passed" : "\(failures) failed")
