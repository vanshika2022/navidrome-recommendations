"""
Enrich Chameleon Swift audio objects with title/artist/track_id metadata.

Two things happen here:

  1. For each audio/<track_id>.mp3 in the Swift container, set Swift
     object metadata (X-Object-Meta-Title, X-Object-Meta-Artist,
     X-Object-Meta-Track-Id) pulled from track_dict.parquet on MinIO.
     These headers show up in the Chameleon Object Details UI and are
     queryable via `swift stat`. Done via POST — no re-upload of audio
     bytes, so it runs fast (~2k objects in a minute or two).

  2. Also write a human-readable manifest.json back to the bucket,
     one entry per track, as a single-file view of the combined data.

Reads track_dict.parquet from MinIO (S3 auth); writes to Chameleon Swift
(Keystone auth). Run on any machine with both sets of credentials.

Setup
-----
    pip install python-swiftclient python-keystoneclient boto3 pyarrow

    # Chameleon Swift (from ~/.chi_auth.sh or equivalent)
    export OS_AUTH_URL=https://chi.uc.chameleoncloud.org:5000/v3
    export OS_AUTH_TYPE=v3applicationcredential
    export OS_APPLICATION_CREDENTIAL_ID=...
    export OS_APPLICATION_CREDENTIAL_SECRET=...

    # MinIO (where track_dict lives)
    export MINIO_URL=http://minio.navidrome-platform.svc.cluster.local:9000
    export MINIO_USER=...
    export MINIO_PASSWORD=...

Usage
-----
    python scripts/make_swift_manifest.py
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pyarrow.parquet as pq
from keystoneauth1.identity.v3 import ApplicationCredential
from keystoneauth1.session import Session
from swiftclient.client import Connection as SwiftConnection
from swiftclient.exceptions import ClientException


log = logging.getLogger("manifest")

_TRACK_ID_RE = re.compile(r"([^/]+)\.mp3$", re.IGNORECASE)


def make_swift() -> SwiftConnection:
    auth_url = os.environ.get("OS_AUTH_URL", "")
    app_id   = os.environ.get("OS_APPLICATION_CREDENTIAL_ID", "")
    app_sec  = os.environ.get("OS_APPLICATION_CREDENTIAL_SECRET", "")
    if not (auth_url and app_id and app_sec):
        raise SystemExit(
            "OS_AUTH_URL + OS_APPLICATION_CREDENTIAL_ID + OS_APPLICATION_CREDENTIAL_SECRET required."
        )
    auth = ApplicationCredential(
        auth_url=auth_url,
        application_credential_id=app_id,
        application_credential_secret=app_sec,
    )
    return SwiftConnection(session=Session(auth=auth))


def make_minio_s3():
    url  = os.environ.get("MINIO_URL", "") or os.environ.get("S3_ENDPOINT_URL", "")
    user = os.environ.get("MINIO_USER", "") or os.environ.get("AWS_ACCESS_KEY_ID", "")
    pw   = os.environ.get("MINIO_PASSWORD", "") or os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    if not url:
        raise SystemExit("MINIO_URL (or S3_ENDPOINT_URL) required.")
    return boto3.client(
        "s3",
        endpoint_url=url,
        aws_access_key_id=user,
        aws_secret_access_key=pw,
        region_name="us-east-1",
    )


def load_track_dict(s3, bucket: str, key: str) -> dict[str, dict]:
    log.info(f"Loading track_dict from s3://{bucket}/{key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    table = pq.read_table(io.BytesIO(obj["Body"].read()))
    tids    = table.column("track_id")
    titles  = table.column("title")
    artists = table.column("artist")
    out: dict[str, dict] = {}
    for i in range(len(tids)):
        out[str(tids[i].as_py())] = {
            "title":  (titles[i].as_py()  or "").strip(),
            "artist": (artists[i].as_py() or "").strip(),
        }
    log.info(f"Loaded metadata for {len(out)} tracks")
    return out


def list_swift_audio(swift: SwiftConnection, container: str, prefix: str) -> list[dict]:
    log.info(f"Listing swift://{container}/{prefix}*")
    entries: list[dict] = []
    marker = ""
    while True:
        _headers, page = swift.get_container(container, prefix=prefix, marker=marker, limit=1000)
        if not page:
            break
        for obj in page:
            if obj["name"].lower().endswith(".mp3"):
                entries.append({
                    "name":  obj["name"],
                    "bytes": obj.get("bytes", 0),
                    "last_modified": obj.get("last_modified"),
                })
        marker = page[-1]["name"]
    log.info(f"Found {len(entries)} mp3s under {container}/{prefix}")
    return entries


def _sanitize_header(value: str) -> str:
    """Swift metadata headers must be ISO-8859-1-encodable; coerce via URL-quote.

    Swift servers vary in their Unicode handling. Safest is to strip anything
    outside ASCII. For demo purposes the title/artist are usually ASCII
    already; the occasional accented character drops but nothing breaks.
    """
    return value.encode("ascii", errors="ignore").decode("ascii").strip()


def update_swift_metadata(swift: SwiftConnection, container: str, obj_name: str,
                          track_id: str, title: str, artist: str) -> None:
    """POST new metadata headers to an existing Swift object (no re-upload).

    Any X-Object-Meta-* header sent REPLACES the set — so we send all three
    in one call and don't have to worry about losing existing ones (there
    aren't any interesting ones on raw mp3s in this bucket).
    """
    headers = {
        "X-Object-Meta-Track-Id": _sanitize_header(track_id),
    }
    if title:
        headers["X-Object-Meta-Title"] = _sanitize_header(title)
    if artist:
        headers["X-Object-Meta-Artist"] = _sanitize_header(artist)
    swift.post_object(container=container, obj=obj_name, headers=headers)


def build_manifest(entries: list[dict], track_dict: dict[str, dict],
                   container: str) -> dict:
    tracks: dict[str, dict] = {}
    missing_meta = 0
    for e in entries:
        m = _TRACK_ID_RE.search(e["name"])
        if not m:
            continue
        track_id = m.group(1)
        meta = track_dict.get(track_id, {})
        if not meta:
            missing_meta += 1
        tracks[track_id] = {
            "track_id":      track_id,
            "title":         meta.get("title", ""),
            "artist":        meta.get("artist", ""),
            "swift_path":    f"swift://{container}/{e['name']}",
            "swift_object":  e["name"],
            "size_bytes":    e["bytes"],
            "last_modified": e["last_modified"],
        }
    return {
        "version":          1,
        "container":        container,
        "track_count":      len(tracks),
        "missing_metadata": missing_meta,
        "tracks":           tracks,
    }


def upload_manifest(swift: SwiftConnection, container: str, key: str, payload: dict) -> None:
    body = json.dumps(payload, indent=2, default=str).encode()
    log.info(f"Uploading manifest → swift://{container}/{key} ({len(body)/1024:.1f} KB)")
    swift.put_object(
        container=container,
        obj=key,
        contents=body,
        content_type="application/json",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--swift-container", default="navidrome-bucket-proj05")
    p.add_argument("--swift-audio-prefix", default="audio/")
    p.add_argument("--swift-manifest-key", default="audio/manifest.json")
    p.add_argument("--minio-meta-bucket", default=os.environ.get("TRACK_META_BUCKET", "navidrome-metadata"))
    p.add_argument("--minio-meta-key",    default=os.environ.get("TRACK_META_KEY", "track_dict.parquet"))
    p.add_argument("--concurrency", type=int, default=8,
                   help="Parallel POSTs when writing object metadata.")
    p.add_argument("--skip-object-metadata", action="store_true",
                   help="Only write the manifest; don't touch per-object X-Object-Meta-* headers.")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, a.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    swift = make_swift()
    s3    = make_minio_s3()

    track_dict = load_track_dict(s3, a.minio_meta_bucket, a.minio_meta_key)
    entries    = list_swift_audio(swift, a.swift_container, a.swift_audio_prefix)
    manifest   = build_manifest(entries, track_dict, a.swift_container)

    log.info(
        f"Manifest: {manifest['track_count']} tracks, "
        f"{manifest['missing_metadata']} missing parquet metadata"
    )

    # Update Swift object metadata in parallel: each audio/<id>.mp3 gets
    # X-Object-Meta-{Title,Artist,Track-Id} set so it shows up in the
    # Chameleon Object Details UI.
    if not a.skip_object_metadata:
        ok = 0
        fail = 0

        def _update(track_id, track_meta):
            nonlocal ok, fail
            try:
                update_swift_metadata(
                    swift,
                    a.swift_container,
                    track_meta["swift_object"],
                    track_id=track_id,
                    title=track_meta.get("title", ""),
                    artist=track_meta.get("artist", ""),
                )
                ok += 1
                if ok % 100 == 0:
                    log.info(f"metadata updated: {ok}/{manifest['track_count']}")
            except ClientException as e:
                fail += 1
                log.warning(f"meta update failed {track_meta['swift_object']}: {e}")

        with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            futs = [pool.submit(_update, tid, tm) for tid, tm in manifest["tracks"].items()]
            for _ in as_completed(futs):
                pass
        log.info(f"Object metadata writes: ok={ok} fail={fail}")

    upload_manifest(swift, a.swift_container, a.swift_manifest_key, manifest)
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
