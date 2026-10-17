"""Merge downloaded Kaggle datasets into the canonical data/ folder.

Handles name normalisation, deduplication (MD5), and corrupt-image filtering.
Designed to be idempotent: running it twice won't create duplicates.

    python -m saabi.merge_datasets
"""
import hashlib
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from . import config

# ── Name normalisation map ──────────────────────────────────────────────
# Keys are the lowercased, stripped folder names found in each source.
# Values are the canonical folder names we use in data/.
BREED_ALIASES = {
    # FarmTrack names
    "sahiwal": "Sahiwal",
    "tharparkar": "Tharparkar",
    "holstein-friesian": "Holstein_Friesian",
    "holstein_friesian": "Holstein_Friesian",
    "holsteinfriesian": "Holstein_Friesian",
    "gir": "Gir",
    "red-sindhi": "Red_Sindhi",
    "red_sindhi": "Red_Sindhi",
    "redsindhi": "Red_Sindhi",
    "rathi": "Rathi",
    "vechur": "Vechur",
    "jersey": "Jersey",
    "ayrshire": "Ayrshire",
    "umblachery": "Umblachery",
    "brown-swiss": "Brown_Swiss",
    "brown_swiss": "Brown_Swiss",
    "brownswiss": "Brown_Swiss",
    "nagori": "Nagori",
    "nili-ravi": "Nili_Ravi",
    "nili_ravi": "Nili_Ravi",
    "niliravi": "Nili_Ravi",
    "jaffarabadi": "Jaffarabadi",
    "jaffrabadi": "Jaffarabadi",
    "kankrej": "Kankrej",
    # User's existing
    "alambadi": "Alambadi",
    "amritmahal": "Amritmahal",
    "dangi": "Dangi",
    "murrah": "Murrah",
    # Possible additions from sujayroy
    "ongole": "Ongole",
    "kangayam": "Kangayam",
    "hallikar": "Hallikar",
    "deoni": "Deoni",
    "bargur": "Bargur",
    "khillari": "Khillari",
    "hariana": "Hariana",
    "punganur": "Punganur",
    "kasaragod": "Kasaragod",
    "kenkatha": "Kenkatha",
    "kherigarh": "Kherigarh",
    "malnad gidda": "Malnad_Gidda",
    "malnad_gidda": "Malnad_Gidda",
    "nimari": "Nimari",
    "pulikulam": "Pulikulam",
    "banni": "Banni",
    "mehsana": "Mehsana",
    "nagpuri": "Nagpuri",
    "shurti": "Shurti",
    "surti": "Surti",
    # Additional breeds from sujayroy dataset
    "red_dane": "Red_Dane",
    "krishna_valley": "Krishna_Valley",
    "guernsey": "Guernsey",
    "toda": "Toda",
    "bhadawari": "Bhadawari",
    "kasargod": "Kasaragod",
    "kasaragod": "Kasaragod",
    "malnad_gidda": "Malnad_Gidda",
    "nili_ravi": "Nili_Ravi",
}


def _canonical_name(raw: str) -> str:
    """Normalise a folder name to its canonical breed name."""
    key = raw.strip().lower().replace(" ", "_")
    if key in BREED_ALIASES:
        return BREED_ALIASES[key]
    # If unknown, title-case with underscores
    return raw.strip().replace("-", "_").replace(" ", "_").title()


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _extract_archive(archive_path: Path, dest: Path) -> Path:
    """Extract a zip archive and return the extracted directory."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest)
    return dest


def _find_breed_dirs(root: Path) -> dict[str, Path]:
    """Walk the extracted directory tree and find breed image directories.

    Handles nested structures like dataset/train/Gir/ or dataset/Gir/images/.
    """
    breed_dirs = {}
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        # Skip common non-breed directories
        if p.name.lower() in {"train", "test", "val", "valid", "validation",
                               "images", ".ipynb_checkpoints", "__macosx",
                               "__pycache__", ".git"}:
            continue
        # A directory is a breed dir if it contains image files directly
        has_images = any(
            f.suffix.lower() in config.IMAGE_EXTENSIONS
            for f in p.iterdir()
            if f.is_file()
        )
        if has_images:
            canonical = _canonical_name(p.name)
            # Prefer the directory with more images if we see the same breed twice
            if canonical in breed_dirs:
                existing_count = sum(1 for f in breed_dirs[canonical].iterdir()
                                     if f.suffix.lower() in config.IMAGE_EXTENSIONS)
                new_count = sum(1 for f in p.iterdir()
                                if f.suffix.lower() in config.IMAGE_EXTENSIONS)
                if new_count <= existing_count:
                    continue
            breed_dirs[canonical] = p
    return breed_dirs


def merge(sources: list[Path], data_dir: Path = config.DATA_DIR,
          min_images: int = 50) -> dict:
    """Merge breed images from multiple sources into data_dir.

    Returns a report with per-breed counts and what was added.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Collect existing hashes from data/ to avoid duplicates
    existing_hashes: dict[str, set[str]] = {}
    for breed_dir in sorted(data_dir.iterdir()):
        if not breed_dir.is_dir() or breed_dir.name.startswith("."):
            continue
        hashes = set()
        for f in breed_dir.iterdir():
            if f.suffix.lower() in config.IMAGE_EXTENSIONS:
                hashes.add(_md5(f))
        existing_hashes[breed_dir.name] = hashes

    report = {"sources": [], "breeds_added": [], "images_added": 0,
              "duplicates_skipped": 0, "corrupt_skipped": 0}

    for source_path in sources:
        source_report = {"path": str(source_path), "breeds": {}}
        breed_dirs = _find_breed_dirs(source_path)

        for canonical, src_dir in sorted(breed_dirs.items()):
            dst_dir = data_dir / canonical
            dst_dir.mkdir(exist_ok=True)
            if canonical not in existing_hashes:
                existing_hashes[canonical] = set()
                report["breeds_added"].append(canonical)

            added = 0
            for img_file in sorted(src_dir.iterdir()):
                if img_file.suffix.lower() not in config.IMAGE_EXTENSIONS:
                    continue
                if not _is_valid_image(img_file):
                    report["corrupt_skipped"] += 1
                    continue
                h = _md5(img_file)
                if h in existing_hashes[canonical]:
                    report["duplicates_skipped"] += 1
                    continue
                existing_hashes[canonical].add(h)
                # Copy with a unique name
                idx = len(list(dst_dir.iterdir()))
                ext = img_file.suffix.lower()
                dst = dst_dir / f"{canonical.lower()}_{idx:04d}{ext}"
                while dst.exists():
                    idx += 1
                    dst = dst_dir / f"{canonical.lower()}_{idx:04d}{ext}"
                shutil.copy2(img_file, dst)
                added += 1

            source_report["breeds"][canonical] = added
            report["images_added"] += added

        report["sources"].append(source_report)

    # Prune breeds with too few images
    pruned = []
    for breed_dir in sorted(data_dir.iterdir()):
        if not breed_dir.is_dir() or breed_dir.name.startswith("."):
            continue
        count = sum(1 for f in breed_dir.iterdir()
                    if f.suffix.lower() in config.IMAGE_EXTENSIONS)
        if count < min_images:
            pruned.append(f"{breed_dir.name} ({count} images)")
            shutil.rmtree(breed_dir)
    report["pruned_breeds"] = pruned

    # Final summary
    final_breeds = {}
    for breed_dir in sorted(data_dir.iterdir()):
        if breed_dir.is_dir() and not breed_dir.name.startswith("."):
            count = sum(1 for f in breed_dir.iterdir()
                        if f.suffix.lower() in config.IMAGE_EXTENSIONS)
            final_breeds[breed_dir.name] = count
    report["final_breeds"] = final_breeds
    report["total_images"] = sum(final_breeds.values())
    return report


def main():
    import json

    cache_base = Path.home() / ".cache" / "kagglehub" / "datasets"
    sources = []

    # Extract downloaded archives
    for dataset_dir in sorted(cache_base.iterdir()):
        for version_dir in sorted(dataset_dir.rglob("*.archive")):
            archive = version_dir
            if archive.exists() and archive.stat().st_size > 0:
                extract_dir = archive.parent / "extracted"
                if not extract_dir.exists():
                    print(f"Extracting {archive.name} ({archive.stat().st_size / 1e6:.0f} MB)...")
                    _extract_archive(archive, extract_dir)
                sources.append(extract_dir)

    if not sources:
        print("No downloaded datasets found in ~/.cache/kagglehub/")
        return

    print(f"Merging from {len(sources)} sources into {config.DATA_DIR}")
    report = merge(sources, min_images=50)

    print(f"\nAdded {report['images_added']} images "
          f"({report['duplicates_skipped']} duplicates, "
          f"{report['corrupt_skipped']} corrupt skipped)")
    if report["breeds_added"]:
        print(f"New breeds: {', '.join(report['breeds_added'])}")
    if report["pruned_breeds"]:
        print(f"Pruned (too few images): {', '.join(report['pruned_breeds'])}")
    print(f"\nFinal dataset: {report['total_images']} images across "
          f"{len(report['final_breeds'])} breeds:")
    for breed, count in sorted(report["final_breeds"].items()):
        bar = "█" * (count // 20)
        print(f"  {breed:<25} {count:>5}  {bar}")

    (config.MODEL_DIR / "merge_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
