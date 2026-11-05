"""Convert a pinned safetensors checkpoint locally; keep only the Q8 serving artifact."""

import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx

from scripts.install.install_sentinel import file_hash, install_runtime
from scripts.runtime.pi_download import download
from velocicyber.engine.sessions import private_write
from velocicyber.models.pi_artifacts import local_file, valid_receipt
from velocicyber.providers.gguf import metadata_from_prefix


def artifact(pin, destination):
    """Download an exact pinned archive, with bounded size and no credential arguments."""
    if destination.is_symlink():
        raise ValueError("Archive must be a regular local file.")
    if destination.exists():
        if destination.stat().st_size == pin["bytes"] and file_hash(destination) == pin["sha256"]:
            return destination
        raise ValueError("Cached converter source checksum mismatch; inspect before retrying.")
    partial = destination.with_suffix(".partial")
    if partial.is_symlink():
        raise ValueError("Partial archive must be a regular local file.")
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
        with client.stream("GET", pin["url"]) as response, partial.open("wb") as output:
            response.raise_for_status()
            for block in response.iter_bytes():
                if output.tell() + len(block) > pin["bytes"]:
                    raise ValueError("Converter archive exceeded its pinned size.")
                output.write(block)
    if partial.stat().st_size != pin["bytes"] or file_hash(partial) != pin["sha256"]:
        raise ValueError("Converter source checksum mismatch.")
    partial.replace(destination)
    return destination


def prepare_converter(cfg, stage, run):
    source = stage / "source"
    # Always re-extract the verified source; an interrupted attempt cannot inject code.
    if source.is_symlink():
        raise ValueError("Converter source directory must not be a symlink.")
    archive = artifact(cfg["conversion"]["source"], stage / "source.tar.gz")
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if sum(m.size for m in members) > 512 * 1024**2:
            raise ValueError("Converter source exceeds its extraction limit.")
        bundle.extractall(source, members=members, filter="data")
    checkout = source / ("llama.cpp-" + cfg["conversion"]["source"]["revision"])
    if not (checkout / "convert_hf_to_gguf.py").is_file():
        raise ValueError("Pinned converter is missing.")
    venv = stage / "venv"
    if venv.is_symlink():
        raise ValueError("Converter environment must not be a symlink.")
    python = venv / "bin/python"
    if not python.exists():
        run([sys.executable, "-m", "venv", str(venv)])
    # This checkpoint's TokenizersBackend needs Transformers 5; the upstream
    # converter requirements still pin 4.x. Keep our compatible pins isolated.
    requirements = Path(__file__).resolve().parents[2] / "config/locks/pi-conversion.requirements.txt"
    if file_hash(requirements) != cfg["conversion"]["requirements_sha256"]:
        raise ValueError("Pi converter dependency pins changed.")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")}
    env["PIP_CONFIG_FILE"] = os.devnull
    run([str(python), "-m", "pip", "install", "--no-cache-dir", "--only-binary=:all:",
         "--timeout", "60", "-r", str(requirements)], cwd=checkout, env=env)
    return python, checkout


def convert(cfg, model_dir, output, python, checkout, run):
    env = {k: v for k, v in os.environ.items()
           if k in {"PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT"}}
    temporary = model_dir.parent / "tmp"
    temporary.mkdir(exist_ok=True)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="3",
               TMPDIR=str(temporary))
    run([str(python), str(checkout / "convert_hf_to_gguf.py"), str(model_dir),
         "--outfile", str(output), "--outtype", cfg["conversion"]["outtype"], "--use-temp-file"],
        cwd=checkout, env=env)
    with output.open("rb") as stream:
        meta = metadata_from_prefix(stream.read(65536))
    if meta.get("general.architecture") != cfg["conversion"]["architecture"]:
        raise ValueError("Converted model architecture does not match the Pi profile.")
    if meta.get("general.file_type") != 7:  # GGUF MOSTLY_Q8_0
        raise ValueError("Converted model is not the configured Q8_0 artifact.")
    return file_hash(output)


def install_converted(cfg, folder, progress):
    if folder.is_symlink():
        raise ValueError("Pi conversion installation must not be a symlink.")
    folder = folder.resolve()
    stage = folder / "conversion-work"
    if stage.is_symlink():
        raise ValueError("Conversion work directory must not be a symlink.")
    receipt_path = folder / "conversion.json"
    if receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if valid_receipt(cfg, folder, saved):
            if stage.exists():
                shutil.rmtree(stage)
            return {**saved, **cfg}
        raise ValueError("Converted Pi installation changed; inspect it before reinstalling.")
    conversion = cfg["conversion"]
    if shutil.disk_usage(folder).free < conversion["minimum_free_gib"] * 1024**3:
        raise RuntimeError(f"Allow {conversion['minimum_free_gib']} GiB free during Pi conversion.")
    from velocicyber.providers.pi import available_memory

    if available_memory() < conversion["minimum_available_gib"] * 1024**3:
        raise RuntimeError("LFM3 conversion needs 6 GiB available RAM. Stop other local models first.")
    stage.mkdir(mode=0o700, exist_ok=True)
    model_dir = stage / "model"
    if model_dir.is_symlink():
        raise ValueError("Conversion model directory must not be a symlink.")
    model_dir.mkdir(exist_ok=True)
    progress("build", "installing_runtime")
    binary, runtime_files = install_runtime(cfg["runtime"], folder)
    for entry in cfg["source_files"]:
        if Path(entry["filename"]).name != entry["filename"] or entry["filename"].endswith(".py"):
            raise ValueError("Only pinned model data files may enter conversion.")
        path = download({"repo": cfg["repo"], "revision": cfg["revision"], **entry}, stage,
                        lambda status, **data: progress("download", status, **data))
        target = model_dir / entry["filename"]
        if target.is_symlink():
            raise ValueError("Conversion input must not be a symlink.")
        if target.exists():
            target.unlink()
        os.link(path, target)
    output = local_file(folder, cfg["filename"])
    partial = output.with_suffix(".partial.gguf")
    if output.exists() or partial.is_symlink():
        raise ValueError("Unverified conversion output exists; inspect it before retrying.")
    partial.unlink(missing_ok=True)
    with (folder / "conversion.log").open("a") as log:
        def run(argv, **kwargs):
            subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT, **kwargs)

        progress("build", "preparing_converter")
        python, checkout = prepare_converter(cfg, stage, run)
        progress("build", "converting")
        checksum = convert(cfg, model_dir, partial, python, checkout, run)
        for source, destination in [(model_dir / "LICENSE", folder / "LICENSE.model"),
                                    (model_dir / "chat_template.jinja", folder / "chat_template.jinja"),
                                    (checkout / "LICENSE", folder / "LICENSE.llama.cpp")]:
            if destination.is_symlink():
                raise ValueError("Installed support file must not be a symlink.")
            shutil.copyfile(source, destination)
    partial.replace(output)
    runtime_stats = {name: [(folder / name).stat().st_size, (folder / name).stat().st_mtime_ns]
                     for name in runtime_files}
    saved = {**cfg, "model_path": str(output), "binary_path": str(binary),
             "converted_weight_bytes": output.stat().st_size, "model_sha256": checksum,
             "model_mtime_ns": output.stat().st_mtime_ns,
             "runtime_files": runtime_files, "runtime_stats": runtime_stats}
    private_write(receipt_path, json.dumps(saved, indent=2) + "\n")
    # Remove only this installer's temporary source/venv/BF16 cache after success.
    shutil.rmtree(stage)
    progress("download", "ready", bytes=output.stat().st_size, total=output.stat().st_size)
    return saved
