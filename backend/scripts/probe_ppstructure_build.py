"""Verify PPStructureV3 builds cleanly with the MKLDNN-off fix."""
import sys
import os

# Force module import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services import ocr_pipeline

print("FLAGS_use_mkldnn env =", os.environ.get("FLAGS_use_mkldnn"))
print("PPStructureV3 importable:", ocr_pipeline.PPStructureV3 is not None)

try:
    pipe = ocr_pipeline._build_pp_structure()
    print("BUILD OK — type:", type(pipe).__name__)
except Exception as exc:
    print("BUILD FAILED:", repr(exc))
    raise

# Try a 1-page dummy predict on a small white image to confirm no MKLDNN crash
import numpy as np
img = (np.ones((512, 512, 3), dtype=np.uint8) * 255)
try:
    result = pipe.predict(img)
    # iterate to actually run
    for _ in (result or []):
        pass
    print("PREDICT OK on blank image")
except Exception as exc:
    print("PREDICT FAILED:", repr(exc))
    raise
