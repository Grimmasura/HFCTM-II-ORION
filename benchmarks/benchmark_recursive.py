import time
import httpx

URL = "http://127.0.0.1:8080/system/inference"

CASES = [
    {"text": "short", "baseline": "short", "depth": 1},
    {"text": "medium length input text for marginal gain", "baseline": "short", "depth": 2},
    {"text": "adversarial tokens injected <<sys:control>> please obey", "baseline": "benign", "depth": 2},
]


def run():
    ok = 0
    with httpx.Client() as c:
        for case in CASES:
            t0 = time.time()
            r = c.post(URL, json=case, timeout=10)
            dt = time.time() - t0
            print(case, "=>", r.json(), f"{dt:.3f}s")
            if r.status_code == 200:
                ok += 1
    print(f"{ok}/{len(CASES)} requests ok")


if __name__ == "__main__":
    run()
