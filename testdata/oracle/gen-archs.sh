#!/bin/sh
# Makes testdata/archs: test-llama-archs's random-weight model for each
# architecture llama.cpp can save (seed 42), and llama.cpp's logits for a
# fixed run of 48 tokens through each (arch_logits.cpp). The models are a
# few MB each and not committed (.gitignore); the logits are, so a model
# remade from the same llama.cpp commit and seed checks against them.
#
#   LLAMA=../llama.cpp BUILD=$LLAMA/build testdata/oracle/gen-archs.sh [arch-regex]
#
# BUILD is a llama.cpp build with tests on (-DLLAMA_BUILD_TESTS=ON).
set -e
: "${LLAMA:?set LLAMA to a llama.cpp checkout}"
: "${BUILD:=$LLAMA/build}"
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../archs"
mkdir -p "$out"
c++ -std=c++17 -O2 -I"$LLAMA/include" -I"$LLAMA/ggml/include" "$here/arch_logits.cpp" \
    -L"$BUILD/bin" -lllama -Wl,-rpath,"$BUILD/bin" -o "$out/.arch_logits"
"$BUILD/bin/test-llama-archs" -s 42 -a "${1:-.*}" -o "$out" > "$out/.gen.log" 2>&1
for m in "$out"/*.gguf; do
    name=$(basename "$m" .gguf)
    if "$out/.arch_logits" "$m" 48 > "$out/$name.logits.tmp" 2>> "$out/.gen.log"; then
        mv "$out/$name.logits.tmp" "$out/$name.logits"
    else
        rm -f "$out/$name.logits.tmp"
        echo "no logits for $name" >&2
    fi
done
git -C "$LLAMA" rev-parse --short HEAD > "$out/LLAMA_COMMIT"
