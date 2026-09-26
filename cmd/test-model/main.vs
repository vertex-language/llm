// The facade: choosing a format from a Hub file list (model/fetch),
// finding a family, loading local references, and refusing what no family
// claims (llm).
package main

import (
    "fs"
    "gpu"
    "llm"
    "model"
    "model/fetch"
    "remote/hub"
)

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

func files(_ paths: [string]) -> [hub.RepoFile] {
    return paths.map { hub.RepoFile(path: $0, size: 1, sha256: "x") }
}

// Choosing a format.
let meta = files(["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "model.safetensors.index.json", "pytorch_model.bin", "onnx/model.onnx", "original/consolidated.00.pth",
                  "tokenizer.json", "tokenizer.model", "tokenizer_config.json"])
let c1 = try fetch.Choose(meta, ref: try hub.Ref.Parse("hf.co/meta-llama/Llama-2-7b-hf"))
check(c1.Format == .safetensors, "a snapshot with config.json and safetensors: safetensors")
var fetched: [string] = []
for f in meta where c1.Include.contains(where: { hub.Match($0, f.Path) }) { fetched.append(f.Path) }
check(fetched == ["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                  "model.safetensors.index.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json"],
      "only the top-level config, weights and tokenizer are fetched: \(fetched)")
let ggufOnly = files(["README.md", "Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf"])
check(try fetch.Choose(ggufOnly, ref: try hub.Ref.Parse("hf.co/unsloth/Qwen3-0.6B-GGUF")).Format == .gguf, "a repository of GGUF files: GGUF")
check(try fetch.Choose(meta, ref: try hub.Ref.Parse("hf.co/x/y:Q4_K_M")).Format == .gguf, "a quant tag: GGUF")
check(try fetch.Choose(meta, ref: try hub.Ref.Parse("hf.co/x/y/model.gguf")).Format == .gguf, "a .gguf file named: GGUF")
do {
    _ = try fetch.Choose(files(["README.md", "pytorch_model.bin", "config.json"]), ref: try hub.Ref.Parse("hf.co/x/old"))
    check(false, "PyTorch weights alone are refused")
} catch {
    check("\(error)".contains("neither safetensors"), "PyTorch weights alone are refused: \(error)")
}

// References and families.
check(fetch.IsLocal("./x") && fetch.IsLocal("/tmp") && fetch.IsLocal("testdata/tiny-llama") && !fetch.IsLocal("hf.co/a/b") && !fetch.IsLocal("a/b"),
      "local paths and Hub references told apart")
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
