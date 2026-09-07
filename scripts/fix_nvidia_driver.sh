#!/usr/bin/env bash
# Repair a broken NVIDIA DKMS install on Ubuntu, of the kind that leaves
# `nvidia-smi` unable to talk to the driver and `torch.cuda.is_available()`
# False on a machine that does have a GPU.
#
# The situation it fixes: a DKMS nvidia module was built for an older kernel,
# the machine now boots a newer one, the module fails to rebuild there, and
# that failure also breaks the postinst of linux-image-*, leaving dpkg
# half-configured so every later `apt` run errors out.
#
# What it does, in order:
#   1. removes the nvidia DKMS modules that are not built for the running kernel
#   2. purges the packages belonging to those versions (and any leftover
#      removed-but-not-purged nvidia packages)
#   3. runs `dpkg --configure -a` to unbreak the interrupted transaction
#   4. installs the newest available open-flavour driver (Hopper and later data
#      centre GPUs need the open kernel modules; the proprietary ones are not
#      supported there from R560 on)
#   5. checks the module actually built for the running kernel
#
# It never reboots and never touches GRUB.
#
# Usage:
#   bash scripts/fix_nvidia_driver.sh                 # dry run: only prints
#   sudo bash scripts/fix_nvidia_driver.sh --yes      # actually do it
#   sudo bash scripts/fix_nvidia_driver.sh --yes --package nvidia-driver-590-server-open
#
# Run scripts/diagnose_gpu.sh first if you have not: this script assumes the
# GPU is visible on the PCI bus.

APPLY=0
PACKAGE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --yes|-y)      APPLY=1 ;;
        --package|-p)  shift; PACKAGE="$1" ;;
        --help|-h)     sed -n '2,32p' "$0"; exit 0 ;;
        *)             echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

have() { command -v "$1" >/dev/null 2>&1; }
say()  { printf '\n== %s ==\n' "$1"; }

run() {
    if [ "$APPLY" = "1" ]; then
        echo "+ $*"
        "$@" || { echo "FAILED: $*" >&2; return 1; }
    else
        echo "would run: $*"
    fi
}

# ── preconditions ─────────────────────────────────────────────────────────
if ! have dkms || ! have apt-get || ! have dpkg; then
    echo "This script targets Ubuntu/Debian with dkms installed." >&2
    echo "Missing one of: dkms, apt-get, dpkg." >&2
    exit 1
fi

if [ "$APPLY" = "1" ] && [ "$(id -u)" != "0" ]; then
    echo "--yes needs root: sudo bash $0 --yes" >&2
    exit 1
fi

RUNNING_KERNEL=$(uname -r)
echo "running kernel : $RUNNING_KERNEL"
if [ -d "/lib/modules/$RUNNING_KERNEL/build" ]; then
    echo "kernel headers : present"
else
    echo "kernel headers : MISSING - install linux-headers-$RUNNING_KERNEL first," \
         "no DKMS module can build without them"
fi
[ "$APPLY" = "1" ] || echo "mode           : DRY RUN (nothing is changed; re-run with --yes)"

# ── 1. stale DKMS modules ─────────────────────────────────────────────────
say "DKMS state"
dkms status 2>/dev/null || true

NVIDIA_VERSIONS=$(dkms status 2>/dev/null | sed -n 's|^nvidia/\([^,]*\),.*|\1|p' | sort -u)
STALE=""
for version in $NVIDIA_VERSIONS; do
    if dkms status 2>/dev/null | grep -q "^nvidia/$version, *$RUNNING_KERNEL"; then
        echo "nvidia/$version is already built for $RUNNING_KERNEL - keeping it"
    else
        STALE="$STALE $version"
    fi
done

say "Removing DKMS modules not built for $RUNNING_KERNEL"
if [ -z "$STALE" ]; then
    echo "(none)"
else
    for version in $STALE; do
        echo "stale: nvidia/$version"
        run dkms remove "nvidia/$version" --all
    done
fi

# ── 2. purge the packages behind them ─────────────────────────────────────
say "Purging their packages"
PURGE=""
for version in $STALE; do
    major=${version%%.*}
    matches=$(dpkg -l 2>/dev/null | awk '/^[a-z][a-z] +nvidia/ {print $2}' \
              | grep -E -- "-${major}(-|\$)" || true)
    PURGE="$PURGE $matches"
done
# packages already removed but still holding config files (dpkg state "rc"):
# they contribute nothing and can block a clean reinstall of the same version
LEFTOVER=$(dpkg -l 2>/dev/null | awk '/^rc +nvidia/ {print $2}' || true)
PURGE=$(echo "$PURGE $LEFTOVER" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')

if [ -z "$(echo "$PURGE" | tr -d ' ')" ]; then
    echo "(nothing to purge)"
else
    echo "packages: $PURGE"
    # shellcheck disable=SC2086
    run apt-get purge -y $PURGE
fi

# ── 3. unbreak dpkg ───────────────────────────────────────────────────────
say "Repairing the interrupted dpkg transaction"
run dpkg --configure -a
run apt-get -f install -y

# ── 4. install the driver ─────────────────────────────────────────────────
say "Choosing a driver package"
if [ -z "$PACKAGE" ]; then
    for pattern in '^nvidia-driver-[0-9]+-server-open$' \
                   '^nvidia-driver-[0-9]+-open$' \
                   '^nvidia-driver-[0-9]+-server$'; do
        PACKAGE=$(apt-cache search --names-only "$pattern" 2>/dev/null \
                  | awk '{print $1}' | sort -V | tail -1)
        [ -n "$PACKAGE" ] && break
    done
fi

if [ -z "$PACKAGE" ]; then
    echo "No nvidia-driver package found in the configured repositories." >&2
    echo "Run 'sudo apt update' first, or pass one explicitly with --package." >&2
    exit 1
fi
echo "installing: $PACKAGE"
echo "(open flavour is required on Hopper and later data centre GPUs)"
run apt-get install -y "$PACKAGE"

# ── 5. verify ─────────────────────────────────────────────────────────────
say "Result"
dkms status 2>/dev/null || true

if [ "$APPLY" != "1" ]; then
    echo
    echo "Dry run finished. Re-run with: sudo bash $0 --yes"
    exit 0
fi

if dkms status 2>/dev/null | grep -q "^nvidia/.*, *$RUNNING_KERNEL.*installed"; then
    cat <<MSG

The module built for $RUNNING_KERNEL. Reboot, then check:

  sudo reboot
  nvidia-smi
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

If nvidia-smi works but torch reports False, the torch build is CPU-only:

  pip install --force-reinstall torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
MSG
else
    BUILD_LOG=$(ls -t /var/lib/dkms/nvidia/*/build/make.log 2>/dev/null | head -1)
    cat <<MSG

The module still did not build for $RUNNING_KERNEL - this driver does not
support that kernel. Read the failure:

  tail -40 ${BUILD_LOG:-/var/lib/dkms/nvidia/<version>/build/make.log}

The pragmatic way out is to boot an older kernel the driver does support
("Advanced options for Ubuntu" in GRUB) and re-run this script there; DKMS
builds for whichever kernel is running. Installed kernels:

$(dpkg -l 2>/dev/null | awk '/^ii +linux-image-[0-9]/ {print "  " $2}')
MSG
fi
