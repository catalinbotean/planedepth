#!/usr/bin/env bash
# Read-only diagnostic for "nvidia-smi has failed / torch.cuda.is_available() is False".
#
# Changes nothing on the machine: it only reads PCI, kernel module, package and
# dmesg state, then prints a verdict with the command that would fix it.
#
# Usage:
#   bash scripts/diagnose_gpu.sh
#
# sudo is used only for dmesg (and only if already available without a
# password); everything else runs unprivileged.

section() { printf '\n== %s ==\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

RUNNING_KERNEL=$(uname -r)

# ── raw state ─────────────────────────────────────────────────────────────
section "SYSTEM"
uname -a
[ -r /etc/os-release ] && . /etc/os-release && echo "distro: ${PRETTY_NAME:-unknown}"
if have systemd-detect-virt; then
    echo "virtualisation: $(systemd-detect-virt 2>&1)"
fi

section "PCI"
if have lspci; then
    PCI_NVIDIA=$(lspci -nn 2>/dev/null | grep -iE "nvidia" || true)
    lspci -nn 2>/dev/null | grep -iE "nvidia|vga compatible|3d controller" || echo "(no display controller found)"
else
    PCI_NVIDIA=""
    echo "lspci not installed (sudo apt install pciutils)"
fi

section "KERNEL MODULES"
MODULES=$(lsmod 2>/dev/null | grep -E "^(nvidia|nouveau)" || true)
[ -n "$MODULES" ] && echo "$MODULES" || echo "(neither nvidia nor nouveau is loaded)"

section "DEVICE NODES"
ls -l /dev/nvidia* 2>&1 || true

section "DRIVER PACKAGES"
PKGS=""
if have dpkg; then
    PKGS=$(dpkg -l 2>/dev/null | grep -iE "nvidia-driver|nvidia-dkms|nvidia-utils|cuda-drivers" \
        | awk '{print $1, $2, $3}')
elif have rpm; then
    PKGS=$(rpm -qa 2>/dev/null | grep -i nvidia)
fi
[ -n "$PKGS" ] && echo "$PKGS" || echo "(no nvidia driver package installed)"

section "DKMS"
if have dkms; then
    dkms status 2>&1 || true
else
    echo "(dkms not installed)"
fi

section "SECURE BOOT"
if have mokutil; then
    mokutil --sb-state 2>&1 || true
else
    echo "(mokutil not installed)"
fi

section "KERNEL HEADERS"
if [ -d "/lib/modules/$RUNNING_KERNEL/build" ]; then
    echo "headers present for running kernel $RUNNING_KERNEL"
else
    echo "MISSING headers for running kernel $RUNNING_KERNEL"
fi

section "NVIDIA-SMI"
if have nvidia-smi; then
    nvidia-smi 2>&1 | head -15
else
    echo "(nvidia-smi not installed)"
fi

section "TORCH"
if have python; then
    python - <<'PY' 2>&1
try:
    import torch
    print("torch      :", torch.__version__)
    print("cuda build :", torch.version.cuda)
    print("available  :", torch.cuda.is_available())
    print("device cnt :", torch.cuda.device_count())
except Exception as exc:
    print("could not import torch:", exc)
PY
else
    echo "(no python on PATH - activate the conda env first)"
fi

section "DMESG"
DMESG=""
if sudo -n true 2>/dev/null; then
    DMESG=$(sudo dmesg 2>/dev/null | grep -iE "nvrm|nvidia|nouveau|vfio" | tail -25 || true)
else
    DMESG=$(dmesg 2>/dev/null | grep -iE "nvrm|nvidia|nouveau|vfio" | tail -25 || true)
fi
if [ -n "$DMESG" ]; then
    echo "$DMESG"
else
    echo "(nothing, or dmesg needs sudo: run 'sudo dmesg | grep -i nvrm | tail -25')"
fi

# ── verdict ───────────────────────────────────────────────────────────────
section "VERDICT"

NVIDIA_LOADED=$(echo "$MODULES" | grep -c "^nvidia " || true)
NOUVEAU_LOADED=$(echo "$MODULES" | grep -c "^nouveau" || true)
DRIVER_PKG=""
have dpkg && DRIVER_PKG=$(dpkg -l 2>/dev/null | grep -cE "^ii +nvidia-driver-[0-9]+" || true)
SB_STATE=""
have mokutil && SB_STATE=$(mokutil --sb-state 2>/dev/null | grep -ci enabled || true)
DKMS_OTHER_KERNEL=""
have dkms && DKMS_OTHER_KERNEL=$(dkms status 2>/dev/null | grep -i nvidia | grep -vc "$RUNNING_KERNEL" || true)

if [ -z "$PCI_NVIDIA" ]; then
    cat <<'MSG'
No NVIDIA device on the PCI bus of this machine.

Inside a VM this means the GPU is not passed through: the guest genuinely has
no card, and nothing installed here can fix it. It has to be configured on the
host (bind the GPU to vfio-pci, host must not be using it). Modern GPUs also
generally need machine type q35 + OVMF/UEFI rather than the i440FX/SeaBIOS
default. On a cloud VM it means the instance type has no GPU attached.
MSG
elif [ "$NOUVEAU_LOADED" != "0" ]; then
    cat <<'MSG'
The open-source `nouveau` driver holds the card, so the NVIDIA kernel module
cannot bind to it. Blacklist it and rebuild the initramfs:

  echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
  sudo update-initramfs -u && sudo reboot
MSG
elif [ "$DRIVER_PKG" = "0" ] || [ -z "$DRIVER_PKG" ]; then
    cat <<'MSG'
The card is visible on PCI but no nvidia-driver package is installed - only the
userspace/CUDA bits. Install the kernel driver (>= 525 for the cu121 wheels
this repo pins) and reboot:

  sudo ubuntu-drivers devices          # see what is recommended
  sudo apt install -y nvidia-driver-535
  sudo reboot
MSG
elif echo "$DMESG" | grep -qi "API mismatch"; then
    cat <<'MSG'
dmesg reports an NVRM API mismatch: the loaded kernel module and the userspace
libraries come from different driver versions (typical after a partial
upgrade). Reinstall a single consistent version and reboot:

  sudo apt install --reinstall -y nvidia-driver-535 nvidia-dkms-535
  sudo reboot
MSG
elif [ "$SB_STATE" != "0" ] && [ -n "$SB_STATE" ] && [ "$NVIDIA_LOADED" = "0" ]; then
    cat <<'MSG'
Secure Boot is enabled and the nvidia module is not loaded: an unsigned DKMS
module is refused by the kernel. Either enroll the MOK key
(`sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`, then confirm in the
blue screen at the next boot) or disable Secure Boot in the VM firmware.
MSG
elif [ -n "$DKMS_OTHER_KERNEL" ] && [ "$DKMS_OTHER_KERNEL" != "0" ] && [ "$NVIDIA_LOADED" = "0" ]; then
    cat <<MSG
The DKMS module is built for a different kernel than the running one
($RUNNING_KERNEL) - it was not rebuilt after a kernel update. Rebuild and
reboot:

  sudo apt install --reinstall -y nvidia-dkms-535 linux-headers-$RUNNING_KERNEL
  sudo reboot
MSG
elif [ "$NVIDIA_LOADED" = "0" ]; then
    cat <<'MSG'
The driver is installed but the module is not loaded. Try loading it by hand -
the error it prints is usually conclusive:

  sudo modprobe nvidia; echo "exit=$?"; sudo dmesg | tail -15
MSG
else
    cat <<'MSG'
The nvidia kernel module is loaded. If torch still reports available: False,
the torch build itself is CPU-only (check "cuda build" above - None or a
version ending in +cpu). Reinstall the CUDA wheels this repo pins:

  pip install --force-reinstall torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
MSG
fi

echo
echo "Paste this whole output if you want help reading it."
