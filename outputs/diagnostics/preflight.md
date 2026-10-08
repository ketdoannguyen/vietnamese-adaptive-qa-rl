# Preflight report

Generated: 2026-10-08 11:02:14 UTC

**Status:** `READY_FOR_TRAIN`

## Checks

- PASS — `gpu_present`
- PASS — `gpu_vram_ge_20gb`
- PASS — `torch_cuda_available`
- PASS — `ram_ge_12gb`
- PASS — `disk_ok`
- PASS — `hf_reachable`
- PASS — `pypi_reachable`

## GPUs

| idx | name | total MiB | free MiB | driver | cc |
|---|---|---|---|---|---|
| 0 | NVIDIA L4 | 23034 | 22564 | 580.178.04 | 8.9 |

## torch

```json
{
  "version": "2.14.1+cu130",
  "cuda_runtime": "13.0",
  "cuda_available": true,
  "device_count": 1,
  "device_name": "NVIDIA L4",
  "total_memory_gib": 22.03,
  "capability": "8.9",
  "usable_bf16": true
}
```

## System

```json
{
  "hostname": "gpu-1-day",
  "user": "nkd",
  "platform": "Linux-6.8.0-1069-gcp-x86_64-with-glibc2.35",
  "kernel": "6.8.0-1069-gcp",
  "cpu_count": 4,
  "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz",
  "memory_mib": {
    "MemTotal": 15990,
    "MemAvailable": 12582,
    "SwapTotal": 0
  },
  "disks": {
    "root": {
      "path": "/",
      "total_gb": 77.3,
      "free_gb": 50.3,
      "used_pct": 35.0
    },
    "home": {
      "path": "/home/nkd",
      "total_gb": 77.3,
      "free_gb": 50.3,
      "used_pct": 35.0
    },
    "tmp": {
      "path": "/tmp",
      "total_gb": 77.3,
      "free_gb": 50.3,
      "used_pct": 35.0
    }
  },
  "python": "3.10.12",
  "venv": null,
  "cwd": "/home/nkd/training-code/vietnamese-adaptive-qa-rl"
}
```

## Network

```json
{
  "huggingface": {
    "url": "https://huggingface.co",
    "http_code": "200",
    "ok": true
  },
  "pypi": {
    "url": "https://pypi.org",
    "http_code": "200",
    "ok": true
  },
  "github": {
    "url": "https://github.com",
    "http_code": "200",
    "ok": true
  }
}
```

## Disk guard

```json
{
  "free_gb": 50.3,
  "min_free_gb": 20.0,
  "ok": true
}
```
