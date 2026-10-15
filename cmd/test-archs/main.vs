// Every family against llama.cpp, architecture by architecture:
// test-llama-archs's random-weight models (testdata/archs, made by
// testdata/oracle/gen-archs.sh) through whichever family claims each, on
// every device, compared with llama.cpp's logits for the same 48 tokens
// by normalized mean squared error -- test-llama-archs's own measure, and
// its bound of 1e-4.
//
// Each model runs three ways, and each must match: the tokens as one
// batch, every row's logits wanted; 10 as a batch and 6 one at a time; and
// two sequences side by side in one memory. An architecture no family
// claims yet is listed as missing, and one a family claims but refuses
// (a feature not run yet) as to do -- not failed: the lists are what is
// left.
//
//   vsc run test-archs [arch-substring]
package main

import (
    "fs"
    "gpu"
    "llm"
    "llm/arch"
    "model"
)

var failures = 0

func check(_ ok: bool, _ what: string) {
    print(ok ? "ok    \(what)" : "FAIL  \(what)")
    if !ok { failures += 1 }
}

// nmse is Σ(a - b)² / Σb²: b the reference.
func nmse(_ a: [float32], _ b: [float32]) -> float64 {
    var num: float64 = 0, den: float64 = 0
    for i in 0..<min(a.count, b.count) {
        let d = float64(a[i]) - float64(b[i])
        num += d * d
        den += float64(b[i]) * float64(b[i])
    }
    if a.count != b.count { return float64.infinity }
    return den > 0 ? num / den : num
}

func int32At(_ b: [uint8], _ at: int) -> int32 {
    return int32(bitPattern: uint32(b[at]) | uint32(b[at + 1]) << 8 | uint32(b[at + 2]) << 16 | uint32(b[at + 3]) << 24)
}

let dir = fs.Path("testdata/archs")
let filter = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ""
var names: [string] = []
for e in (try? fs.ReadDir(dir)) ?? [] where e.Name.hasSuffix(".logits") {
    let name = String(e.Name.dropLast(7))
    if filter.isEmpty || name.contains(filter) {
        names.append(name)
    }
}
names.sort()
if names.isEmpty {
    print("no fixtures in testdata/archs: run testdata/oracle/gen-archs.sh")
}
var missing: [string] = []
var todo: [string] = []
var passed: [string] = []
for name in names {
    let gg = dir / (name + ".gguf")
    if !fs.Exists(gg) {
        print("skip  \(name): no \(name).gguf (testdata/oracle/gen-archs.sh makes it)")
        continue
    }
    let raw = try fs.ReadFile(dir / (name + ".logits"))
    let n = int(int32At(raw, 4)), vocab = int(int32At(raw, 8))
    var tokens: [int] = []
    for i in 0..<n { tokens.append(int(int32At(raw, 12 + 4 * i))) }
    var want: [float32] = []
    let at = 12 + 4 * n
    for i in 0..<(n * vocab) {
        want.append(float32(bitPattern: uint32(bitPattern: int32At(raw, at + 4 * i))))
    }
    let ck = try model.Open(gg)
    if llm.FamilyOf(ck) == nil {
        missing.append(name)
        continue
    }
    var ok = true
    for d in gpu.Devices() {
        var loaded: (any arch.Model)? = nil
        do {
            loaded = try await llm.Load(ck, on: d)
        } catch {
            todo.append("\(name): \(error)")
        }
        guard let m = loaded else {
            ok = false
            break
        }
        // One batch, every row's logits.
        let all = try await m.Decode(arch.Batch.Prompt(tokens, all: true), arch.Output.logits).Download()
        let e1 = nmse(all, want)
        // 10 as a batch, then 6 a token at a time.
        var split = try await m.Decode(arch.Batch.Prompt(Array(tokens[0..<10]), all: true), arch.Output.logits).Download()
        for i in 10..<n {
            split += try await m.Decode(arch.Batch.Prompt([tokens[i]], from: i), arch.Output.logits).Download()
        }
        let e2 = nmse(split, want)
        // Two sequences in one batch: the first 8 of sequence 1 and all of
        // sequence 0 (each from position 0, which drops what it held).
        var e3: float64 = 0
        do {
            var b = arch.Batch.Prompt(Array(tokens[0..<8]), sequence: 1, all: true)
            let b0 = arch.Batch.Prompt(tokens, sequence: 0, all: true)
            b.Tokens += b0.Tokens
            b.Positions += b0.Positions
            b.Sequences += b0.Sequences
            b.Outputs += b0.Outputs
            let both = try await m.Decode(b, arch.Output.logits).Download()
            e3 = max(nmse(Array(both[0..<(8 * vocab)]), Array(want[0..<(8 * vocab)])), nmse(Array(both[(8 * vocab)...]), want))
        }
        let good = e1 <= 1e-4 && e2 <= 1e-4 && e3 <= 1e-4
        check(good, "\(name) on \(d.Name): nmse \(e1) batch, \(e2) split, \(e3) two sequences")
        ok = ok && good
    }
    if ok { passed.append(name) }
}
print("\(passed.count) of \(names.count) fixtures match llama.cpp; \(todo.count) claimed but not run yet; \(missing.count) have no family yet")
for t in todo {
    print("todo  \(t)")
}
if !missing.isEmpty {
    print("missing: \(missing.joined(separator: " "))")
}
print(failures == 0 ? "all passed" : "\(failures) failed")
