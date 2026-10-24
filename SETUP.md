# Setup checklist: Mini LLM Inference Engine

## Already done

- Project folder `D:\mini-infer` with the repo layout from the plan
- Virtual env `.venv` (Python 3.11.9)
- PyTorch with CUDA 12.8 plus everything in `requirements.txt`

Activate it (PowerShell):

```powershell
D:\mini-infer\.venv\Scripts\Activate.ps1
```

To recreate it from scratch:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

---

## Working in VS Code

Open the folder `D:\mini-infer` itself (File → Open Folder), not `D:\`. The `.vscode/` settings then:
- select `.venv` as the interpreter and activate it in new terminals
- put the project root on `PYTHONPATH`, so `import engine...` works everywhere
- cap GPU memory at 75% (`MINI_INFER_GPU_FRACTION`) for anything run from the terminal
- add Run and Debug entries (F5): baseline server, load test, current file
- enable the Testing panel (pytest, `tests/`)

Install the recommended extensions when VS Code prompts (Python, Pylance, Python Debugger).

## Week 0 commands (run from `D:\mini-infer`)

```powershell
# 1. Build frozen workloads (already done: sharegpt_1000, synthetic_p128_o64_n64)
python -m bench.make_workloads sharegpt
python -m bench.make_workloads synthetic --prompt-len 512 --output-len 128 -n 200

# 2. Start the HF baseline (terminal 1). 1 = sequential, N = static batching
python -m server.baseline_server --max-batch-size 8

# 3. Load test it (terminal 2). Results land in bench/results/
python -m bench.load_test --workload bench/workloads/sharegpt_1000.jsonl --num-requests 200 --concurrency 32 --repeats 3 --name hf_static_bs8
```

GPU memory: the server caps PyTorch at 75% of the card (`--gpu-fraction 0.6` to go lower).
This caps memory, not compute: while a benchmark runs the GPU will be busy, which is what you want for honest numbers.

---

## Accounts and API keys you need

| # | What | Why | Where | Cost |
|---|------|-----|-------|------|
| 1 | **Hugging Face account + access token** (Read scope) | Download Qwen2.5 weights and datasets; needed for gated datasets | huggingface.co → Settings → Access Tokens | Free |
| 2 | **Accept LMSYS-Chat-1M license** on HF | The second workload is a gated dataset | huggingface.co/datasets/lmsys/lmsys-chat-1m → "Agree" | Free |
| 3 | **Weights & Biases account + API key** | Experiment tracking | wandb.ai/authorize (apply for the free academic plan) | Free for students |
| 4 | **GitHub account** | Repo, CI (GitHub Actions), optionally GitHub Pages | github.com | Free |
| 5 | **Kaggle account** (phone-verify it to get GPUs) | Free T4/P100 GPU time, about 30 h/week | kaggle.com → Settings | Free |
| 6 | **Google account** for Colab | Backup free GPU (T4) | colab.research.google.com | Free (Pro is optional) |
| 7 | **GPU rental account** (pick one: RunPod, Lambda, Vast.ai, Modal) | A few hours on one A10G/L4/A100 for the final benchmarks (week 5) | Their sites | About $1 to 4 per hour; budget $10 to 20 |
| 8 | **Vercel account** (or use GitHub Pages) | Host the project website | vercel.com | Free |
| 9 | *(Optional)* HF Space | Live demo on the 0.5B model | huggingface.co/new-space | Free on CPU; GPU Spaces cost money |

Put the keys in a `.env` file (copy `.env.example`). Never commit it; `.gitignore` already excludes it.
You can also log in once from the CLI instead:

```powershell
huggingface-cli login
wandb login
```

---

## Software to install on Windows (outside the venv)

| Tool | Why | Status / how |
|------|-----|-----|
| NVIDIA driver | GPU | Installed (610.47) |
| Python 3.11 | Plan specifies 3.11 | Installed |
| Git | Version control, tags each week | Installed (2.53) |
| Node.js | Astro website + Vercel CLI (week 5) | Installed (v24) |
| VS Code (or your editor) | Editing and debugging | Optional |
| **OBS Studio** or ScreenToGif | 60-second demo video / GIF | `winget install OBSProject.OBSStudio` |
| Vercel CLI | Deploy the site | `npm i -g vercel` after Node is installed |
| *(Stretch)* Triton | Fused paged-attention kernel | Official Triton has no Windows build. Use `pip install triton-windows` or do kernel work in WSL2/Kaggle/Linux |
| *(Optional)* WSL2 + Ubuntu | Linux path for Triton and vLLM comparison runs | `wsl --install` |

You do **not** need a separate CUDA Toolkit install. The PyTorch wheels bundle the CUDA runtime. You'd only need the toolkit to compile custom CUDA C++ extensions, and this plan doesn't.

---

## Model weights and datasets (download from HF, about 5 GB total)

- `Qwen/Qwen2.5-0.5B-Instruct` (about 1 GB). Use it for all development
- `Qwen/Qwen2.5-1.5B-Instruct` (about 3 GB). Final benchmarks and the speculative-decoding verifier
- `anon8231489123/ShareGPT_Vicuna_unfiltered`: main load test
- `lmsys/lmsys-chat-1m` (gated): second workload
- `wikitext` (`wikitext-2-raw-v1`): INT8 perplexity
- `cais/mmlu` or `Rowan/hellaswag`: accuracy checks

Tip: set `HF_HOME=D:\hf-cache` so the large downloads stay on D: and not in your user folder on C:.

---

## Note on your GPU (RTX 3050 Laptop, 6 GB)

This is below the plan's suggested 8 GB, but it's workable:
- **0.5B in FP16** (about 1 GB of weights) leaves plenty of room for KV cache. Do all development on it locally.
- **1.5B in FP16** (about 3 GB) fits, but leaves less KV room, so you'll hit the concurrency limit sooner.
- **Speculative decoding (0.5B + 1.5B together)** is about 4 GB of weights and tight. It's fine for correctness work; get real numbers on Kaggle or the rented GPU.
- Final headline benchmarks should come from one rented GPU anyway (methodology rule: one consistent GPU).
