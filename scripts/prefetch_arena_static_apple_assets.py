#!/usr/bin/env python3
"""Download the complete USD dependency trees used by the static apple demo."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

from isaaclab.utils import assets as asset_utils

ASSETS = {
    "Galileo background": (
        "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
        "IsaacLab/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd"
    ),
    "G1 robot": (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
        "Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd"
    ),
    "Apple": (
        "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
        "IsaacLab/Arena/assets/object_library/srl_robolab_assets/objects/objaverse/apple_01.usd"
    ),
    "Plate": (
        "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/"
        "IsaacLab/Arena/assets/object_library/srl_robolab_assets/objects/hot3d/clay_plates.usd"
    ),
}


def _remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=30) as response:
        return int(response.headers["Content-Length"])


def _download_range(url: str, path: Path, start: int, end: int) -> None:
    expected_size = end - start + 1
    current_size = path.stat().st_size if path.exists() else 0
    if current_size == expected_size:
        return
    if current_size > expected_size:
        path.unlink()
        current_size = 0

    for attempt in range(8):
        range_start = start + current_size
        request = urllib.request.Request(url, headers={"Range": f"bytes={range_start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Server ignored byte range for {url}: HTTP {response.status}"
                    )
                with path.open("ab") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            current_size = path.stat().st_size
            if current_size == expected_size:
                return
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            current_size = path.stat().st_size if path.exists() else 0
            if attempt == 7:
                raise RuntimeError(f"Failed range {range_start}-{end} for {url}") from exc
            time.sleep(min(2**attempt, 10))

    raise RuntimeError(f"Incomplete range {start}-{end} for {url}")


def _download_file(url: str, target: Path, workers: int) -> None:
    size = _remote_size(url)
    if target.is_file() and target.stat().st_size == size:
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    part_dir = target.with_name(f".{target.name}.parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    part_count = min(workers, max(1, math.ceil(size / (4 * 1024 * 1024))))
    chunk_size = math.ceil(size / part_count)

    ranges: list[tuple[Path, int, int]] = []
    for index in range(part_count):
        start = index * chunk_size
        end = min(size - 1, start + chunk_size - 1)
        ranges.append((part_dir / f"part-{index:03d}", start, end))

    with ThreadPoolExecutor(max_workers=part_count) as executor:
        futures = {
            executor.submit(_download_range, url, part_path, start, end): index
            for index, (part_path, start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            future.result()

    combined = target.with_name(f".{target.name}.download")
    with combined.open("wb") as output:
        for part_path, _, _ in ranges:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, output, length=1024 * 1024)
    if combined.stat().st_size != size:
        raise RuntimeError(
            f"Size mismatch for {url}: got {combined.stat().st_size}, expected {size}"
        )
    os.replace(combined, target)
    shutil.rmtree(part_dir)


def _cache_target(cache_dir: Path, url: str) -> Path:
    return cache_dir / urlparse(url).path.lstrip("/")


def _download_tree(root_url: str, cache_dir: Path, workers: int) -> Path:
    local_root = _cache_target(cache_dir, root_url)

    # A root layer is usually much larger than its dependencies, so split it
    # across all connections first.  Once its references are known, download
    # independent dependency files concurrently (one connection per file).
    if not local_root.is_file():
        print(f"[prefetch] Fetching {root_url} ({workers} connections max)", flush=True)
    _download_file(root_url, local_root, workers)

    visited: set[str] = {root_url}
    to_visit = {
        resolved
        for reference in asset_utils._find_asset_dependencies(str(local_root))  # noqa: SLF001
        if (resolved := asset_utils._resolve_reference_url(root_url, reference))  # noqa: SLF001
    }

    while to_visit:
        frontier = to_visit - visited
        to_visit = set()
        if not frontier:
            break
        visited.update(frontier)

        download_urls: list[str] = []
        for url in frontier:
            if not asset_utils._UDIM_RE.search(url):  # noqa: SLF001
                download_urls.append(url)
                continue

            for tile in range(1001, 1101):
                tile_url = asset_utils._UDIM_RE.sub(str(tile), url)  # noqa: SLF001
                try:
                    _remote_size(tile_url)
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        break
                    raise
                if tile_url not in visited:
                    download_urls.append(tile_url)
                    visited.add(tile_url)

        def _fetch_dependency(url: str) -> tuple[str, Path] | None:
            target = _cache_target(cache_dir, url)
            try:
                if not target.is_file():
                    print(f"[prefetch] Fetching {url}", flush=True)
                _download_file(url, target, 1)
            except urllib.error.HTTPError as exc:
                staging_host = "omniverse-content-staging.s3-us-west-2.amazonaws.com"
                production_host = "omniverse-content-production.s3-us-west-2.amazonaws.com"
                if exc.code != 404 or staging_host not in url:
                    print(
                        f"[prefetch] Skipping unavailable dependency ({exc.code}): {url}",
                        flush=True,
                    )
                    return None

                # Arena root layers live on the staging bucket, while shared
                # Isaac materials referenced by those layers can live only in
                # the production bucket.  Mirror either source into the same
                # local path so relative USD references keep working offline.
                fallback_url = url.replace(staging_host, production_host, 1)
                try:
                    print(f"[prefetch] Retrying from production: {fallback_url}", flush=True)
                    _download_file(fallback_url, target, 1)
                except urllib.error.HTTPError as fallback_exc:
                    print(
                        f"[prefetch] Skipping unavailable dependency ({fallback_exc.code}): {url}",
                        flush=True,
                    )
                    return None
                return fallback_url, target
            return url, target

        with ThreadPoolExecutor(max_workers=min(workers, len(download_urls) or 1)) as executor:
            futures = [executor.submit(_fetch_dependency, url) for url in download_urls]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                url, target = result
                for reference in asset_utils._find_asset_dependencies(str(target)):  # noqa: SLF001
                    reference_url = asset_utils._resolve_reference_url(url, reference)  # noqa: SLF001
                    if reference_url and reference_url not in visited:
                        to_visit.add(reference_url)

    return local_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--asset", action="append", choices=ASSETS)
    parser.add_argument("--root-only", action="store_true")
    args = parser.parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    selected_assets = args.asset or list(ASSETS)
    for name in selected_assets:
        url = ASSETS[name]
        print(f"[prefetch] Downloading {name} and dependencies...", flush=True)
        local_path = _cache_target(cache_dir, url)
        if args.root_only:
            if not local_path.is_file():
                print(f"[prefetch] Fetching {url} ({args.workers} connections max)", flush=True)
            _download_file(url, local_path, args.workers)
        else:
            local_path = _download_tree(url, cache_dir, args.workers)
        print(f"[prefetch] Ready: {local_path}", flush=True)

    print(f"[prefetch] All static-apple assets are cached in {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
