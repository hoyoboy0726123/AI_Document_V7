import inspect
from paddleocr import PPStructureV3

sig = inspect.signature(PPStructureV3.__init__)
params = list(sig.parameters.keys())
print("total params:", len(params))

want = [
    "enable_mkldnn",
    "text_detection_model_name",
    "text_recognition_model_name",
    "use_doc_orientation_classify",
    "use_doc_unwarping",
    "use_textline_orientation",
    "cpu_threads",
    "device",
    "layout_detection_model_name",
    "mkldnn",
    "use_mkldnn",
]
for k in want:
    print(f"  {k}: {'YES' if k in params else 'missing'}")

print()
print("--- all params ---")
for k in params:
    print("  -", k)
