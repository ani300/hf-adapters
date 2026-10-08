# shellcheck shell=bash
# Sourced by the "Build hf-adapters" step of this action (and by the Prune uv
# cache workflow). Installs hf-adapters + torch-spyre into the project venv,
# taking the registry wheels from a shared uv cache on the storage PVC when it
# has them.
#
# Why: every Spyre job used to download its ~77 registry wheels (100-550 s on a
# slow mirror day). The storage PVC keeps a uv cache of those wheels, filled by
# the Prune uv cache workflow (fill_shared_uv_cache). Jobs only READ it, in one
# offline `uv sync` of the registry packages, and never export it as
# UV_CACHE_DIR. Resolution and every source build stay on the pod-local cache,
# because on a shared cache
#   * uv holds an exclusive per-source lock (sdists-v9/path/<hash of the path>/
#     .lock) for a whole path build, and every pod builds torch-spyre from the
#     same /home/senuser/torch-spyre, so the builds serialize (`Timeout (300s)
#     when waiting for lock`);
#   * a pod waiting on that lock can reuse the wheel another pod just built
#     from a different torch-spyre SHA at the same path (uv keys path builds on
#     pyproject.toml/setup.py mtimes, not sources);
#   * build environments live in <cache>/builds-v0, so on NFS every build
#     first copies torch into NFS;
#   * filling it from many pods makes them wait on uv's per-wheel locks, which
#     NFS hands from waiter to waiter only every ~20-30 s.
# After that, the job runs the same `uv add` / `uv lock` / `uv sync --refresh`
# as before against its pod-local cache, so the venv ends up exactly as uv.lock
# says; the shared read only saves downloads. Any miss or error leaves the rest
# to that unchanged sync.
#
# Inputs (environment):
#   UV_PROJECT_ENVIRONMENT  the project venv (/home/senuser/.venv)
#   UV_GROUPS               space-separated dependency groups
#   SHARED_UV_CACHE_DIR     shared uv cache; empty/missing = pod-local only
#   TORCH_SPYRE_DIR         torch-spyre checkout (default /home/senuser/torch-spyre)
# Run from the hf-adapters checkout.
#
# Timing lines, for parsing CI logs:
#   SETUP-TIMING phase=<phase> seconds=<s> status=<status>

HFA_DEPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

_hfa_now() { echo "${EPOCHREALTIME/./}"; }
_hfa_timing() {  # phase start_us status
  local d=$(( ($(_hfa_now) - $2) / 100000 ))
  printf 'SETUP-TIMING phase=%s seconds=%d.%d status=%s\n' "$1" $((d / 10)) $((d % 10)) "$3"
}

_hfa_group_flags() {
  local g out=""
  for g in $UV_GROUPS; do out="$out --group $g"; done
  echo "$out"
}

# --no-install-package for every lockfile package built from source (the
# project, path, editable and git sources): those never come from the shared
# cache.
_hfa_exclude_flags() {
  "${UV_PROJECT_ENVIRONMENT}/bin/python" -I -c '
import sys, tomllib
lock = tomllib.load(open(sys.argv[1], "rb"))
names = sorted({p["name"] for p in lock.get("package", []) if "registry" not in p.get("source", {})})
print(" ".join("--no-install-package " + n for n in names))' "$1"
}

# Offline sync of the registry packages from the shared cache, from a copy of
# the committed pyproject.toml/uv.lock so it can run while `uv add` rewrites
# them. With --offline a fully cached sync takes no lock there but the shared
# cache-root lock. Copy mode: the venv is on another filesystem.
_hfa_prefill_from_shared_cache() {
  local proj=$1 t excl
  t=$(_hfa_now)
  excl=$(_hfa_exclude_flags "$proj/uv.lock") || { _hfa_timing shared-sync "$t" error; return 1; }
  # shellcheck disable=SC2086
  if UV_CACHE_DIR="$SHARED_UV_CACHE_DIR" UV_OFFLINE=1 UV_LINK_MODE=copy \
     UV_LOCK_TIMEOUT="${SHARED_UV_LOCK_TIMEOUT:-120}" \
     uv sync --project "$proj" --frozen --no-install-project $excl $(_hfa_group_flags); then
    _hfa_timing shared-sync "$t" ok
  else
    echo "::notice::the shared uv cache lacks wheels this lockfile pins (or is locked); the pod-local sync downloads them"
    _hfa_timing shared-sync "$t" miss
    return 1
  fi
}

_hfa_discard() {
  [[ -n "$1" && -e "$1" ]] || return 0
  if [[ -n "${HFA_TRASH_DIR:-}" ]]; then
    mv -- "$1" "$HFA_TRASH_DIR/$(basename -- "$1").$RANDOM"
  else
    # Detached from the step's output, so the runner does not wait for it.
    rm -rf -- "$1" </dev/null >/dev/null 2>&1 &
  fi
  return 0
}

# Install into $UV_PROJECT_ENVIRONMENT. Same commands as before; the only
# difference is that the venv may already hold the registry packages.
install_hf_adapters() {
  local ts_dir="${TORCH_SPYRE_DIR:-/home/senuser/torch-spyre}"
  local group_flags t_total t proj="" pid="" plog="" status=off
  group_flags=$(_hfa_group_flags)
  t_total=$(_hfa_now)

  if [[ -n "${SHARED_UV_CACHE_DIR:-}" && -d "$SHARED_UV_CACHE_DIR" ]]; then
    proj=$(mktemp -d)
    cp pyproject.toml uv.lock "$proj/"
    plog=$(mktemp)
    _hfa_prefill_from_shared_cache "$proj" >"$plog" 2>&1 &
    pid=$!
  fi

  t=$(_hfa_now)
  uv add --no-sync torch-spyre "$ts_dir"
  _hfa_timing uv-add "$t" ok
  t=$(_hfa_now)
  uv lock --upgrade-package torch-spyre
  _hfa_timing uv-lock "$t" ok

  if [[ -n "$pid" ]]; then
    t=$(_hfa_now)
    if wait "$pid"; then status=hit; else status=miss; fi
    # Only uv's summary lines; the full per-package output is noise here.
    grep -vE '^ [+-] ' "$plog" || true
    rm -f "$plog"
    _hfa_timing shared-wait "$t" "$status"
    _hfa_discard "$proj"
  fi

  # Unchanged from before: builds torch-spyre (and hf-adapters) from source
  # on the pod-local cache, downloads whatever is still missing, and makes the
  # venv match uv.lock exactly.
  t=$(_hfa_now)
  uv sync --frozen --verbose --refresh $group_flags
  _hfa_timing final-sync "$t" "$status"
  _hfa_timing setup-total "$t_total" "$status"
}

# Fill the shared cache with every registry wheel this checkout's uv.lock pins
# (all groups), from a single process. Cheap when everything is cached.
fill_shared_uv_cache() {
  local t base_py tmp excl
  t=$(_hfa_now)
  mkdir -p "$SHARED_UV_CACHE_DIR"
  excl=$(_hfa_exclude_flags uv.lock)
  # The Spyre jobs' interpreter, so the cached wheels have their tags.
  base_py=$("${UV_PROJECT_ENVIRONMENT}/bin/python" -I -c 'import os, sys; print(os.path.realpath(sys._base_executable))')
  tmp=$(mktemp -d)
  # shellcheck disable=SC2086
  UV_CACHE_DIR="$SHARED_UV_CACHE_DIR" UV_PROJECT_ENVIRONMENT="$tmp/venv" UV_LINK_MODE=copy \
    uv sync --quiet --frozen --no-install-project $excl --all-groups --python "$base_py"
  _hfa_discard "$tmp"
  _hfa_timing shared-fill "$t" ok
}
