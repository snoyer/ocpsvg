import logging
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlretrieve

from ocpsvg import ColorAndLabel, import_svg_document


@contextmanager
def timer(title: str):
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        raise
    finally:
        t1 = time.perf_counter()
        logging.info(f"{title}: {t1-t0:0.3f}s")


logging.basicConfig(level=logging.DEBUG)


TIGER_URL = "https://upload.wikimedia.org/wikipedia/commons/f/fd/Ghostscript_Tiger.svg"
tiger_path = Path(tempfile.gettempdir()) / "Ghostscript_Tiger.svg"

if not tiger_path.is_file():
    urlretrieve(TIGER_URL, tiger_path)
if not tiger_path.is_file():
    raise IOError(f"could not retrieve {TIGER_URL}")


with timer(f"import {tiger_path}"):
    imported = list(import_svg_document(tiger_path, metadata=ColorAndLabel))

logging.info(
    " ".join(
        f"{v}*{k.__qualname__}"
        for k, v in Counter(map(type, (o for o, _ in imported))).items()
    )
)
try:
    for face_or_wire, color_and_label in imported:
        show_object(  # type: ignore
            face_or_wire,
            f"{color_and_label.label} #{id(face_or_wire):x}",
            dict(color=color_and_label.color_for(face_or_wire)),
        )
except NameError:  # no `show_object`
    pass
