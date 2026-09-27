"""Standalone performance probe for the face pipeline.

Answers "is my low FPS from frame decode, GPU inference, or lock contention?"
by measuring each stage in isolation:

  1. YOLOv8 detect/track latency per synthetic 1080p frame
  2. InsightFace (antelopev2) detect+embed latency per person crop
  3. Concurrency test: N threads on ONE shared session behind a lock vs
     one session per thread - shows whether cross-camera serialization
     is costing throughput
  4. Optional live RTSP decode rate (--source rtsp://...)

Usage:
    python benchmark.py [--frames 200] [--threads 3] [--source rtsp://user:pass@ip/stream]
"""

import argparse
import logging
import os
import threading
import time

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)


def make_synthetic_frame(width=1920, height=1080):
    """Noise background with a person-ish blob; realistic enough for timing"""
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (width // 2 - 150, height // 2 - 400),
                  (width // 2 + 150, height + 200), (140, 130, 120), -1)
    return frame


def pick_yolo_model():
    for path in ("models/yolov8n.onnx", "models/yolov8n.pt"):
        if os.path.exists(path):
            return path
    raise FileNotFoundError("No yolov8n model found in models/")


def bench_yolo(frames, device):
    from ultralytics import YOLO
    model = YOLO(pick_yolo_model(), task='detect', verbose=False)
    frame = make_synthetic_frame()
    logging.info("Warming up YOLO...")
    for _ in range(10):
        model.track(frame, classes=[0], persist=True,
                    tracker="bytetrack.yaml", device=device, verbose=False)

    times = []
    for i in range(frames):
        t0 = time.perf_counter()
        model.track(frame, classes=[0], persist=True,
                    tracker="bytetrack.yaml", device=device, verbose=False)
        times.append(time.perf_counter() - t0)
    report("YOLO track (1080p, single stream)", times)


def bench_face(crop, frames, providers):
    from insightface.app import FaceAnalysis
    handler = FaceAnalysis('antelopev2', providers=providers, root='.')
    handler.prepare(ctx_id=0)

    logging.info("Warming up InsightFace...")
    for _ in range(5):
        handler.get(crop)

    times = []
    for _ in range(frames):
        t0 = time.perf_counter()
        handler.get(crop)
        times.append(time.perf_counter() - t0)
    report("InsightFace get() per person crop", times)


def bench_concurrency(crop, calls_per_thread, threads, providers):
    """Shared session + lock vs dedicated session per thread."""
    from insightface.app import FaceAnalysis

    def run_shared(handler, lock):
        def worker():
            for _ in range(calls_per_thread):
                with lock:
                    handler.get(crop)
        return worker

    def run_dedicated():
        handler = FaceAnalysis('antelopev2', providers=providers, root='.')
        handler.prepare(ctx_id=0)

        def worker():
            for _ in range(calls_per_thread):
                handler.get(crop)
        return worker

    # Shared
    shared_handler = FaceAnalysis(
        'antelopev2', providers=providers, root='.')
    shared_handler.prepare(ctx_id=0)
    lock = threading.Lock()
    barrier = threading.Barrier(threads)
    results = {}

    def shared_worker(idx):
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(calls_per_thread):
            with lock:
                shared_handler.get(crop)
        results[idx] = time.perf_counter() - t0

    thread_objs = [threading.Thread(target=shared_worker, args=(i,))
                   for i in range(threads)]
    t0 = time.perf_counter()
    for t in thread_objs:
        t.start()
    for t in thread_objs:
        t.join()
    shared_wall = time.perf_counter() - t0
    total_calls = threads * calls_per_thread
    logging.info(f"SHARED session x{threads} threads: {total_calls} calls in "
                 f"{shared_wall:.2f}s -> {total_calls/shared_wall:.1f} calls/s "
                 f"(worst thread waited {max(results.values()):.2f}s)")

    if len(results) != threads:
        return

    # Dedicated
    try:
        workers = [run_dedicated() for _ in range(threads)]
        barrier = threading.Barrier(threads)
        results.clear()

        def dedicated_worker(idx, w):
            barrier.wait()
            t0 = time.perf_counter()
            for _ in range(calls_per_thread):
                w()
            results[idx] = time.perf_counter() - t0

        thread_objs = [threading.Thread(target=dedicated_worker, args=(i, workers[i]))
                       for i in range(threads)]
        t0 = time.perf_counter()
        for t in thread_objs:
            t.start()
        for t in thread_objs:
            t.join()
        dedicated_wall = time.perf_counter() - t0
        logging.info(f"DEDICATED sessions x{threads} threads: {total_calls} calls in "
                     f"{dedicated_wall:.2f}s -> {total_calls/dedicated_wall:.1f} calls/s "
                     f"(worst thread {max(results.values()):.2f}s)")
        gain = (shared_wall / dedicated_wall - 1) * 100
        logging.info(f"Dedicated sessions are {gain:+.0f}% throughput vs shared")
    except Exception as e:
        logging.warning(f"Dedicated-session test skipped ({e})")


def bench_rtsp(source, seconds=15):
    from olderTests.newcamera import FreshestFrame
    logging.info(f"Probing RTSP decode rate for {seconds}s: {source}")
    fresh = FreshestFrame(source)
    seq_start = fresh.latestnum
    time.sleep(seconds)
    decoded = fresh.latestnum - seq_start
    fps = decoded / seconds
    logging.info(f"RTSP decode: {decoded} frames in {seconds}s = {fps:.1f} FPS")
    if fps < 15:
        logging.warning("Decode rate is LOW - network/camera side issue, not your pipeline")
    fresh.release()


def report(label, times_ms_seconds):
    arr = np.array(times_ms_seconds) * 1000
    logging.info(
        f"{label}: avg {arr.mean():.1f}ms | median {np.median(arr):.1f}ms | "
        f"p95 {np.percentile(arr, 95):.1f}ms | max {arr.max():.1f}ms | "
        f"~{1000.0/max(arr.mean(), 1e-6):.1f} FPS equivalent")


def main():
    # Silence per-session "Applied providers" spam from ORT/insightface
    for noisy in ('insightface', 'onnxruntime', 'ultralytics'):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Face pipeline benchmark")
    parser.add_argument("--frames", type=int, default=200,
                        help="iterations per single-stream benchmark")
    parser.add_argument("--threads", type=int, default=3,
                        help="concurrent threads for the session-contention test")
    parser.add_argument("--source", type=str, default=None,
                        help="optional RTSP url to measure decode FPS")
    args = parser.parse_args()

    import torch
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda \
        else ['CPUExecutionProvider']
    logging.info(f"Device: {device}")

    frame = make_synthetic_frame()
    # Person-sized crop like the ones queued by process_frame
    crop = frame[280:880, 810:1110]

    bench_yolo(args.frames, device)
    bench_face(crop, args.frames, providers)
    bench_concurrency(crop, max(10, args.frames // 4), args.threads, providers)
    if args.source:
        bench_rtsp(args.source)


if __name__ == "__main__":
    main()
