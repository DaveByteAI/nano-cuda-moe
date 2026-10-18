#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 for the dsv41 path: acrogenesis' IQ2_XXS-experts / Q8-dense GGUF, 10 files, 359 GB.
#   scripts/download_dsv41.sh [main | engram | all]       (default main)
#     main    files 1-7 and 10 (150 GB): experts, attention, embeddings, output. Into $MODEL_DIR.
#     engram  files 8 and 9 (209 GB): the two Engram n-gram tables, read a few rows per token. Into $ENGRAM_DIR,
#             which can be a slower disk; the engine finds them through a symlink in $MODEL_DIR.
# Resumable, SHA-256 checked after each file (SKIP_SHA=1 skips it).
#
# Speed: with aria2c installed it asks the mirror once per file for the redirect to the HF CDN (a signed URL, valid
# about an hour) and opens 16 connections to the CDN. The mirror answers 429 to 16 connections of its own, and the
# CDN does not; an expired URL is fetched again and the download resumes. Measured 34 MB/s against ~6 MB/s for one
# connection. Without aria2c it falls back to curl. The proxy variables are ignored on the mirror path (unset).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR=${MODEL_DIR:-models/DSV41-IQ2_XXS}
ENGRAM_DIR=${ENGRAM_DIR:-$MODEL_DIR}
ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
REPO=acrogenesis/DeepSeek-V4.1-Flash-1017-IQ2_XXS-GGUF
REV=f5221b0be8e1849905de308e5ab4e73e282ddb60
P=DeepSeek-V4.1-Flash-1017-IQ2_XXS-Experts-Q8-Dense-Engram
WHAT=${1:-main}
FILES=(   # part, size, SHA-256 at REV
  "00001 21027092128 5eeb4867ffa53d23210395cc9f69423769a27fdcf3fc7dcadaab74c435aa01b7"
  "00002 21021328800 03f9d07360b343ad7ce35bea591ac627dfe223d99c82bd4af49a2260d70f3820"
  "00003 21021328832 621859c563fec81156e1898ca77c9967263d0a3af0b6b344cb16ae227510ca8d"
  "00004 21021328832 5ab7bdbe5ec949630094fe9e7a22a8039df3c160820a538a0d8493c4f92a4e58"
  "00005 21021328832 8a0c65d6bb76625c342cf62a95d74e7ca266282376ccb07aeeb1ec6004ba6306"
  "00006 21021328832 7eede4801336ac5700fe226b2b6d21907412f05c5803d91c63d8ac7c44bac4a2"
  "00007 14014219264 c4a96e9547e88f113e3184a470e2041687f4e9f9a64a7b03185c3a74cd4556fb"
  "00008 104449677888 096b13ad2b1875e043053d68cf4cd2cc1f54f4f5df9b07629c4f90be7b776559"
  "00009 104452537696 cdcd2cdd1657cd60354fdb63dc899731665448ed53229dfdd32dd2ca1d517945"
  "00010 10268402880 cb9325b45fc4c585caa83c24f2f6985b7a872ec0546e4fe5af98456631d0c33c"
)
is_engram() { [ "$1" = 00008 ] || [ "$1" = 00009 ]; }

fetch() {   # fetch <url> <dir> <name>
  local url=$1 dir=$2 name=$3
  if ! command -v aria2c >/dev/null; then
    curl -fL --retry 10 --retry-delay 5 -C - -o "$dir/$name" "$url"; return
  fi
  local tries=0
  while :; do
    # one request to the mirror for the CDN location; the CDN takes the 16 connections
    # the mirror itself answers 429 to a burst of requests: never hand aria2c the mirror URL, wait for a location
    local cdn="" wait=10
    while [ -z "$cdn" ]; do
      # || true: a timed-out curl fails the pipeline, and under set -e that would end the script silently
      cdn=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
            curl -sI -m 30 "$url" | awk 'tolower($1)=="location:"{print $2}' | tr -d '\r' | tail -1) || true
      if [ -z "$cdn" ]; then echo "== $name: no redirect from the mirror (rate limit?), again in $wait s"; sleep $wait; wait=$((wait < 120 ? wait * 2 : 120)); fi
    done
    if env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
       aria2c --no-conf -d "$dir" -o "$name" -x16 -s16 -k20M -c --file-allocation=falloc \
              --max-tries=5 --retry-wait=5 --timeout=60 --summary-interval=60 --console-log-level=warn "$cdn"; then
      return
    fi
    tries=$((tries + 1)); [ $tries -lt 20 ] || { echo "$name: giving up after $tries rounds"; return 1; }
    echo "== $name: aria2c stopped (an expired URL or the network), resolving again in 10 s"; sleep 10
  done
}

mkdir -p "$MODEL_DIR" "$ENGRAM_DIR"
for f in "${FILES[@]}"; do
  read -r part size sha <<<"$f"
  if is_engram "$part"; then [ "$WHAT" = main ] && continue; dir=$ENGRAM_DIR
  else [ "$WHAT" = engram ] && continue; dir=$MODEL_DIR; fi
  name="$P-$part-of-00010.gguf"
  out="$dir/$name"
  have=$( [ -f "$out" ] && [ ! -f "$out.aria2" ] && stat -L -c %s "$out" || echo 0 )
  if [ "$have" != "$size" ]; then
    echo "== $name ($((size / 1000000000)) GB) -> $dir"
    fetch "$ENDPOINT/$REPO/resolve/$REV/$name" "$dir" "$name"
  fi
  [ "$(stat -L -c %s "$out")" = "$size" ] || { echo "$out: wrong size; delete it and run again"; exit 1; }
  if [ "${SKIP_SHA:-0}" != 1 ]; then
    echo "== checking $name"
    echo "$sha  $out" | sha256sum -c --quiet || { echo "$out: SHA-256 mismatch; delete it and run again"; exit 1; }
  fi
  # the loader opens every part from the first file's directory
  if [ "$dir" != "$MODEL_DIR" ] && [ ! -e "$MODEL_DIR/$name" ]; then ln -s "$(realpath "$out")" "$MODEL_DIR/$name"; fi
done
echo "done: $WHAT"
