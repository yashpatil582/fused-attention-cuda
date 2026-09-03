#!/usr/bin/env bash
# Track B PROBE 4 — VM ncu go/no-go, run from the Mac BEFORE any project code (judge_plan §2.3).
#
#   usage: gate_gpu.sh [ssh-host]      (default host alias: gpu, from ~/.ssh/config)
#
# PASS = a metric value prints (non-root or sudo). If only sudo works, print NVIDIA's
# documented permanent fix (NVreg_RestrictProfilingToAdminUsers=0 + initramfs + reboot).
# If it fails even under sudo on a real VM: terminate and move to the next provider —
# never debug a host you do not control. Time box: 15 minutes.
set -uo pipefail
HOST=${1:-gpu}

echo "=== PROBE 4a: environment on $HOST ==="
ssh "$HOST" 'nvidia-smi --query-gpu=name,driver_version,compute_cap,clocks.max.sm --format=csv; nvcc --version | tail -2; command -v ncu nsys compute-sanitizer; grep -i RmProfilingAdminOnly /proc/driver/nvidia/params; grep -c . /proc/1/cgroup; sudo -n true && echo SUDO_OK; python3 -c "import torch;print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability())"' 2>&1 | tee /tmp/fa_gate_env.txt

echo; echo "=== PROBE 4b: one-metric ncu as user, then sudo ==="
ssh "$HOST" 'printf "__global__ void k(float*x){x[threadIdx.x]*=2.f;}\nint main(){float*d;cudaMalloc(&d,4096);k<<<1,256>>>(d);cudaDeviceSynchronize();return 0;}\n" > /tmp/t.cu && nvcc -O2 -lineinfo -arch=native /tmp/t.cu -o /tmp/t && ncu --metrics sm__cycles_elapsed.avg /tmp/t; echo "nonroot_rc=$?"; sudo $(which ncu) --metrics sm__cycles_elapsed.avg /tmp/t; echo "root_rc=$?"' 2>&1 | tee /tmp/fa_gate_ncu.txt

echo; echo "=== PROBE 4c: nsys + persistence mode + clocks ==="
ssh "$HOST" 'nsys profile -t cuda -o /tmp/ts /tmp/t >/dev/null 2>&1 && ls /tmp/ts.nsys-rep; sudo nvidia-smi -pm 1 && nvidia-smi -q -d CLOCK | head -20' 2>&1 | tee /tmp/fa_gate_nsys.txt

echo; echo "=== VERDICT ==="
NONROOT=$(grep -oE 'nonroot_rc=[0-9]+' /tmp/fa_gate_ncu.txt | cut -d= -f2); ROOT=$(grep -oE 'root_rc=[0-9]+' /tmp/fa_gate_ncu.txt | tail -1 | cut -d= -f2)
# A metric value line looks like: "sm__cycles_elapsed.avg   cycle   123,456" — that is the only proof.
if grep -qE 'sm__cycles_elapsed\.avg +[a-z]* +[0-9]' /tmp/fa_gate_ncu.txt; then
  if [ "${NONROOT:-1}" = 0 ]; then
    echo "PASS: ncu counters readable as a regular user (rc=0). Proceed with scripts/sync.sh."
  else
    echo "PASS (sudo only): non-root rc=${NONROOT:-?}, root rc=${ROOT:-?}. Make it permanent, then re-run this gate:"
    echo "  ssh $HOST \"echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo update-initramfs -u -k all && sudo reboot\""
    echo "  (NVIDIA ERR_NVGPUCTRPERM page: https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)"
  fi
  exit 0
fi
echo "FAIL: no metric value even under sudo (non-root rc=${NONROOT:-?}, root rc=${ROOT:-?})."
grep -iE 'ERR_NVGPUCTRPERM|permission|not found|No such' /tmp/fa_gate_ncu.txt | head -3
echo "Terminate this instance and try the next provider (AWS -> Lambda -> Vast VM mode); do not debug the host."
exit 1
