#!/usr/bin/env bash
# Downloads the GitHub release files that the deps download when they compile, and gives them
# to the deps through their environment variables. A file that is already in ~/release-files
# (from the cache) does not download again. The deps download with httpc, and the
# redirect of GitHub has a header longer than the limit of httpc (header_too_long). curl does
# not have this limit. Run this script after mix deps.get and before the deps compile.
set -euo pipefail

case "$(uname -s)" in
  Linux)
    xla=x86_64-linux-gnu-cpu
    nif=x86_64-unknown-linux-gnu
    make_nif=x86_64-linux-gnu
    mlx=mlx-x86_64-linux-gnu
    ;;
  Darwin)
    xla=aarch64-darwin-cpu
    nif=aarch64-apple-darwin
    make_nif=aarch64-apple-darwin
    # The macOS archives of MLX have the deployment target of emlx in their name.
    mlx=mlx-arm64-apple-darwin-$(sed -n 's/^  @macos_deployment_target_latest "\(.*\)"$/\1/p' deps/emlx/mix.exs)
    ;;
esac

dir="$HOME/release-files"
mkdir -p "$dir"

version() {
  sed -n "s/^  \"$1\": {:hex, :$1, \"\([^\"]*\)\".*/\1/p" mix.lock
}

# fetch URL SHA256: downloads URL into $dir when it is not there, and checks its SHA-256.
fetch() {
  file="$dir/$(basename "$1")"
  [ -f "$file" ] || curl -fsSL --retry 5 --retry-all-errors -o "$file" "$1"
  sum=$(shasum -a 256 "$file" | cut -d ' ' -f 1)
  if [ "$sum" != "$2" ]; then
    echo "::error::$1 has SHA-256 $sum, expected $2"
    exit 1
  fi
}

# xla, tokenizers (rustler_precompiled) and stb_image (elixir_make) list the SHA-256 of each
# release file in a checksum file. This takes the files of the target of the runner.
for spec in "xla elixir-nx/xla $xla" "tokenizers elixir-nx/tokenizers $nif" \
  "stb_image elixir-nx/stb_image $make_nif"; do
  read -r dep repo target <<< "$spec"
  grep -h -- "-$target[-.]" deps/"$dep"/checksum* | while read -r line; do
    file=$(grep -o '[^" ]*\.tar\.gz' <<< "$line")
    sum=$(grep -o '[0-9a-f]\{64\}' <<< "$line")
    fetch "https://github.com/$repo/releases/download/v$(version "$dep")/$file" "$sum"
  done
done

# emlx has the SHA-256 of the MLX archive in a file next to the archive.
url="https://github.com/cocoa-xu/mlx-build/releases/download/v$(sed -n 's/^  @mlx_version "\(.*\)"$/\1/p' deps/emlx/mix.exs)/$mlx.tar.gz"
fetch "$url" "$(curl -fsSL --retry 5 --retry-all-errors "$url.sha256" | cut -d ' ' -f 1)"

{
  echo "XLA_ARCHIVE_PATH=$dir/xla_extension-$(version xla)-$xla.tar.gz"
  echo "MLX_ARCHIVE_PATH=$dir/$mlx.tar.gz"
  echo "RUSTLER_PRECOMPILED_GLOBAL_CACHE_PATH=$dir"
  echo "ELIXIR_MAKE_CACHE_DIR=$dir"
} >> "$GITHUB_ENV"
