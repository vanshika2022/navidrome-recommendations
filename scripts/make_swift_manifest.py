"""
Generate + upload audio metadata manifest to Chameleon Swift.

Reads track_dict.parquet from MinIO (S3 auth), lists the audio container
on Chameleon Swift (Keystone auth), cross-references to find which
track_ids actually have audio uploaded, then writes a JSON manifest with
{track_id, title, artist, swift_path, size} and uploads it back to Swift
at navidrome-bucket-proj05/audio/manifest.json.

Written so Salauat (or anyone with the Chameleon credentials) can run it
on a machine with both cluster-internal MinIO access and Chameleon Swift
access. The serving container + Navidrome don't depend on this file —
it's purely for human inspection of what metadata the audio bucket has.

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

import boto3
import pyarrow.parquet as pq
from keystoneauth1.identity.v3 import ApplicationCredential
from keystoneauth1.session import Session
from swiftclient.client import Connection as SwiftConnection


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

    upload_manifest(swift, a.swift_container, a.swift_manifest_key, manifest)
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
