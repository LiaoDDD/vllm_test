import os
import ssl
import json
import time
import math
import argparse
import asyncio
from statistics import mean
import httpx
from typing import List, Dict

DEFAULT_BASE_URL = "https://n8n-dev.liontravel.com/n8n/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"

def percentile(data: List[float], p: float) -> float:
    if not data:
        return float("nan")
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data_sorted[int(k)]
    d0 = data_sorted[int(f)] * (c - k)
    d1 = data_sorted[int(c)] * (k - f)
    return d0 + d1

def gen_prompt(n_chars: int) -> str:
    base = "你是一位專業的旅遊文案作家。你的任務是根據使用者提供的旅遊目的地，創作一段簡潔、引人入勝的宣傳文案。請著重描述該地點的獨特魅力、氛圍或必訪景點，並激發讀者的旅行慾望。請不要提供任何額外的解釋或說明，只回傳文案本身。一定要用繁體中文回答。"
    s = (base * ((n_chars // len(base)) + 1))[:n_chars]
    return s

async def one_request(client: httpx.AsyncClient, url: str, model: str,
                      prompt_chars: int, max_tokens: int, api_key: str):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": gen_prompt(prompt_chars)}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t0 = time.perf_counter()
    try:
        r = await client.post(f"{url}/chat/completions", json=payload, headers=headers)
        elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        ok = (r.status_code == 200) and (r.json().get("choices") is not None)
        return ok, elapsed, r.status_code, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000.0
        return False, elapsed, None, str(e)

async def worker(name: int, client: httpx.AsyncClient, url: str, model: str,
                 prompt_chars: int, max_tokens: int, deadline: float,
                 latencies: List[float], codes: Dict[int, int], errors: List[str],
                 think_ms: int):
    while time.time() < deadline:
        ok, elapsed, code, err = await one_request(client, url, model, prompt_chars, max_tokens, os.getenv("VLLM_API_KEY", ""))
        if ok:
            latencies.append(elapsed)
        else:
            latencies.append(elapsed)  # 也記錄失敗延遲
        if code is not None:
            codes[code] = codes.get(code, 0) + 1
        if err:
            errors.append(err)
        # 模擬人類思考間隔，避免完全同步觸發
        await asyncio.sleep(think_ms / 1000.0)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Base URL (…/v1)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--users", type=int, default=100, help="同時使用者數 (VUs)")
    parser.add_argument("--duration", default="180s", help="壓測時間，例：180s / 3m")
    parser.add_argument("--prompt-chars", type=int, default=256, help="單次提問大約字元數（越大 prefill 壓力越高）")
    parser.add_argument("--max-tokens", type=int, default=128, help="回覆的最長 token 數（越大 decode 壓力越高）")
    parser.add_argument("--ramp", action="store_true", help="啟用階梯式加壓 (25→50→75→100)")
    parser.add_argument("--insecure", action="store_true", help="忽略 TLS 憑證驗證（僅測路由時可臨時用）")
    parser.add_argument("--keepalive", type=int, default=100, help="連線池大小")
    parser.add_argument("--timeout", type=float, default=180.0, help="單請求逾時秒數")
    parser.add_argument("--think-ms", type=int, default=0, help="每次請求後的思考時間(ms)")

    args = parser.parse_args()

    # 解析 duration
    dur = 0
    if args.duration.endswith("s"):
        dur = int(args.duration[:-1])
    elif args.duration.endswith("m"):
        dur = int(args.duration[:-1]) * 60
    else:
        dur = int(args.duration)

    # TLS 驗證選項
    verify = True
    if args.insecure:
        verify = False

    limits = httpx.Limits(max_keepalive_connections=args.keepalive, max_connections=args.keepalive)
    timeout = httpx.Timeout(connect=args.timeout, read=args.timeout, write=args.timeout, pool=args.timeout)

    latencies: List[float] = []
    codes: Dict[int, int] = {}
    errors: List[str] = []

    async with httpx.AsyncClient(http2=False, verify=verify, limits=limits, timeout=timeout) as client:
        start = time.time()
        deadline = start + dur

        tasks = []

        if not args.ramp:
            # 直接 N 個使用者一起跑
            for i in range(args.users):
                tasks.append(asyncio.create_task(
                    worker(i, client, args.base_url, args.model,
                           args.prompt_chars, args.max_tokens, deadline,
                           latencies, codes, errors, args.think_ms)
                ))
            await asyncio.gather(*tasks)
        else:
            # 階梯式：每 45s 再增加 25 個使用者，直到 100
            step = 25
            current = 0
            while time.time() < deadline:
                target = min(args.users, current + step)
                for i in range(current, target):
                    tasks.append(asyncio.create_task(
                        worker(i, client, args.base_url, args.model,
                               args.prompt_chars, args.max_tokens, deadline,
                               latencies, codes, errors, args.think_ms)
                    ))
                current = target
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(45, remaining))
            await asyncio.gather(*tasks, return_exceptions=True)

    end = time.time()
    elapsed_s = end - start
    total = len(latencies)
    succ = sum(v for k, v in codes.items() if k == 200)
    success_rate = (succ / max(1, total)) * 100.0

    p50 = percentile(latencies, 0.50)
    p90 = percentile(latencies, 0.90)
    p95 = percentile(latencies, 0.95)
    p99 = percentile(latencies, 0.99)
    avg = mean(latencies) if latencies else float("nan")
    qps = total / elapsed_s if elapsed_s > 0 else 0.0

    print("\n=== vLLM 壓測結果 ===")
    print(f"Base URL       : {args.base_url}")
    print(f"Model          : {args.model}")
    print(f"Users (VUs)    : {args.users}  | Duration: {dur}s | Ramp: {args.ramp}")
    print(f"Prompt chars   : {args.prompt_chars} | max_tokens: {args.max_tokens}")
    print(f"Requests total : {total} | Success(200): {succ} | Success rate: {success_rate:.2f}%")
    print(f"Latency (ms)   : avg {avg:.1f} | p50 {p50:.1f} | p90 {p90:.1f} | p95 {p95:.1f} | p99 {p99:.1f}")
    print(f"Throughput     : {qps:.2f} req/s")
    print(f"HTTP codes     : {json.dumps(codes, ensure_ascii=False)}")
    if errors:
        print(f"Errors (sample): {errors[:3]}  ... total={len(errors)}")
    print("====================\n")

if __name__ == "__main__":
    asyncio.run(main())
