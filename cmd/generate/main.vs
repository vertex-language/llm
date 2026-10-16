// generate: a model by reference -- a registry name or a local path --
// greedily continuing a prompt.
//
//     vsc run generate -- qwen3:0.6b "Once upon a time"
//     vsc run generate -- testdata/stories15M-q4_0.gguf "Once upon a time" --tokens 64
//     vsc run generate -- qwen3:0.6b --inspect
package main

import (
    "gpu"
    "llm"
)

func main() async -> int32 {
    let args = CommandLine.arguments
    if args.count < 2 {
        print("usage: generate <ref> [prompt] [--tokens N] [--cpu] [--inspect]")
        return 2
    }
    let ref = args[1]
    var prompt = "Once upon a time"
    var n = 32
    var cpu = false
    var inspect = false
    var i = 2
    while i < args.count {
        if args[i] == "--tokens" && i + 1 < args.count {
            n = int(args[i + 1]) ?? n
            i += 1
        } else if args[i] == "--cpu" {
            cpu = true
        } else if args[i] == "--inspect" {
            inspect = true
        } else {
            prompt = args[i]
        }
        i += 1
    }
    do {
        if inspect {
            let info = try await llm.Inspect(ref)
            print("\(info.Architecture) (family \(info.Family ?? "none"), \(info.Format.Name), \(info.Tensors) tensors, \(info.Bytes) bytes)")
            for (k, v) in info.Config { print("  \(k) = \(v)") }
            return 0
        }
        let m = try await llm.Load(ref, on: cpu ? gpu.CPU() : gpu.Default())
        let out = try await llm.Generate(m, prompt, tokens: n)
        print(prompt + m.Tokenizer.Decode(out))
    } catch {
        print("\(error)")
        return 1
    }
    return 0
}
