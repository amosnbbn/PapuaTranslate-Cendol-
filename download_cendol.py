from huggingface_hub import snapshot_download
import os

out_dir = r"C:\ML\models\cendol-mt5-base-inst"
os.makedirs(out_dir, exist_ok=True)

snapshot_download(
    "indonlp/cendol-mt5-base-inst",
    local_dir=out_dir,
    local_dir_use_symlinks=False
)

print("✅ Model sudah tersimpan di:", out_dir)
