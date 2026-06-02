import os
import json
from pathlib import Path

def setup_kaggle_credentials():
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("请提供Kaggle API凭证:")
        username = input("Kaggle用户名: ")
        key = input("Kaggle API Key: ")
        
        credentials = {
            "username": username,
            "key": key
        }
        
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        os.chmod(kaggle_json, 0o600)
        print("Kaggle凭证已保存!")
    else:
        print("Kaggle凭证已存在")

def push_notebook():
    import subprocess
    
    notebook_path = "notebooks/petrochemical_monitoring_kaggle.py"
    
    kernel_metadata = {
        "id": "petrochemical_monitoring",
        "title": "Petrochemical Monitoring System - Deep Learning",
        "code_file": notebook_path,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    metadata_path = Path("kernel-metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(kernel_metadata, f, indent=2)
    
    print("\n推送Notebook到Kaggle...")
    result = subprocess.run(
        ['kaggle', 'kernels', 'push', '-p', '.'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("成功推送到Kaggle!")
        print(result.stdout)
    else:
        print("推送失败:")
        print(result.stderr)
        return False
    
    return True

def main():
    print("=" * 60)
    print("Kaggle Notebook 推送工具")
    print("=" * 60)
    
    setup_kaggle_credentials()
    
    if push_notebook():
        print("\n下一步:")
        print("1. 访问 https://www.kaggle.com/kernels")
        print("2. 找到 'Petrochemical Monitoring System' notebook")
        print("3. 点击 'Run All' 执行")
        print("4. 下载生成的图表文件")
    else:
        print("\n请检查Kaggle API配置后重试")

if __name__ == '__main__':
    main()
