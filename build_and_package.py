import os
import shutil
import subprocess
import sys
import hashlib
import zipfile

def run_command(command, cwd=None):
    print(f"Executing: {command} in {cwd or os.getcwd()}")
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)

def get_checksum(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    base_dir = os.getcwd()
    frontend_dir = os.path.join(base_dir, "frontend")
    backend_dir = os.path.join(base_dir, "backend")
    dist_dir = os.path.join(base_dir, "dist")
    build_dir = os.path.join(base_dir, "build")
    package_name = "AI_Document_V3_Packaged"
    
    # 獲取當前 Python 環境的 Scripts 路徑 (針對 Windows)
    python_bin_dir = os.path.dirname(sys.executable)
    pyinstaller_bin = os.path.join(python_bin_dir, "pyinstaller.exe")
    pip_audit_bin = os.path.join(python_bin_dir, "pip-audit.exe")
    
    # 如果在 venv 中找不到，則嘗試使用全域指令
    if not os.path.exists(pyinstaller_bin): pyinstaller_bin = "pyinstaller"
    if not os.path.exists(pip_audit_bin): pip_audit_bin = "pip-audit"

    print("--- [Step 1] Building Frontend ---")
    run_command("npm install", cwd=frontend_dir)
    run_command("npm run build", cwd=frontend_dir)

    print("--- [Step 2] Moving Frontend Assets ---")
    target_frontend_dist = os.path.join(backend_dir, "frontend_dist")
    if os.path.exists(target_frontend_dist):
        shutil.rmtree(target_frontend_dist)
    shutil.copytree(os.path.join(frontend_dir, "dist"), target_frontend_dist)

    print("--- [Step 3] Security Audit ---")
    print(f"Running {pip_audit_bin}...")
    subprocess.run(f"{pip_audit_bin}", shell=True)

    print("--- [Step 4] Packaging with PyInstaller ---")
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    run_command(f"{pyinstaller_bin} package.spec")

    print("--- [Step 5] Finalizing Package ---")
    exe_path = os.path.join(dist_dir, "AI_Document_V3.exe")
    if not os.path.exists(exe_path):
        print("Error: Executable not found!")
        sys.exit(1)

    # 建立最終壓縮檔
    zip_filename = f"{package_name}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(exe_path, arcname="AI_Document_V3.exe")
        # 包含 .env_example 供使用者參考
        env_example = os.path.join(backend_dir, ".env_example")
        if os.path.exists(env_example):
            zipf.write(env_example, arcname=".env.example")
        
    # 生成校驗檔
    checksum = get_checksum(zip_filename)
    with open("checksum.txt", "w") as f:
        f.write(f"SHA-256 for {zip_filename}: {checksum}\n")

    print(f"\n--- Packaging Successful! ---")
    print(f"Result: {zip_filename}")
    print(f"Checksum: {checksum}")
    print("--- Cleaning up temporary files ---")
    # 清理中間產物
    if os.path.exists(target_frontend_dist):
        shutil.rmtree(target_frontend_dist)
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    
    print("\n[Done] You can now distribute the .zip file.")

if __name__ == "__main__":
    main()
