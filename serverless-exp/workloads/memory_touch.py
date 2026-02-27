import time, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-ms", type=int, default=300)
    ap.add_argument("--size-mb", type=int, default=64)
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()

    n = args.size_mb * 1024 * 1024
    buf = bytearray(n)

    end_t = time.time() + (args.target_ms / 1000.0)
    s = 0

    # loop until target time is reached
    while time.time() < end_t:
        for i in range(0, n, args.stride):
            buf[i] = (buf[i] + 1) & 0xFF
            s += buf[i]

    print(s)

if __name__ == "__main__":
    main()