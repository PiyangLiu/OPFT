import hashlib
import shutil
import urllib.request
from pathlib import Path


URL = "https://github.com/PiyangLiu/OPFT/releases/latest/download/train_data.h5"
SHA256 = "fff52d66f0adef89a4f27fb1b9e1e34ec1c9b3c18ec06900b5e306df6851c3b9"
TARGET = Path(__file__).resolve().parent / "train_data.h5"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if TARGET.exists() and file_sha256(TARGET) == SHA256:
        print(f"Dataset is ready: {TARGET}")
        return

    temporary = TARGET.with_suffix(TARGET.suffix + ".part")
    request = urllib.request.Request(
        URL, headers={"User-Agent": "OPFT-data-downloader"}
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output)
        actual = file_sha256(temporary)
        if actual != SHA256:
            raise RuntimeError(
                f"SHA-256 mismatch: expected {SHA256}, received {actual}"
            )
        temporary.replace(TARGET)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    print(f"Downloaded and verified: {TARGET}")


if __name__ == "__main__":
    main()
