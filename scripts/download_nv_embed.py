#!/usr/bin/env python3
"""
下载 Hugging Face 模型 nvidia/NV-Embed-v2 到指定本地目录（默认 /model/squirrel/NV-Embed-v2）。
用法示例:
  pip install huggingface-hub
  python scripts/download_nv_embed.py --output-dir /model/squirrel/NV-Embed-v2

脚本会尝试使用环境变量 HUGGINGFACE_HUB_TOKEN 或 HUGGINGFACE_TOKEN 作为访问令牌，
如果模型为公开模型则不需要 token。
"""

import os
import argparse
from huggingface_hub import snapshot_download


def download_model(repo_id: str = "nvidia/NV-Embed-v2", output_dir: str = "/model/squirrel/NV-Embed-v2", revision: str = None, token: str = None):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {repo_id} -> {output_dir}")
    try:
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
    except Exception as e:
        raise RuntimeError(f"下载失败: {e}")

    # 列出下载后的文件用于验证
    files = []
    for root, dirs, filenames in os.walk(output_dir):
        for fn in filenames:
            files.append(os.path.join(root, fn))
    print(f"Downloaded {len(files)} files. Example files:")
    for f in files[:10]:
        print('  ', f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Hugging Face model to local directory")
    parser.add_argument('--repo-id', default='nvidia/NV-Embed-v2', help='Hugging Face repo id')
    parser.add_argument('--output-dir', default='/model/squirrel/NV-Embed-v2', help='Local directory to save the model')
    parser.add_argument('--revision', default=None, help='Revision/branch/commit id (optional)')
    parser.add_argument('--token', default=None, help='Hugging Face access token (optional). If omitted, will use HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN env vars.')
    args = parser.parse_args()

    token = args.token or os.environ.get('HUGGINGFACE_HUB_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')

    try:
        download_model(repo_id=args.repo_id, output_dir=args.output_dir, revision=args.revision, token=token)
        print('模型下载完成')
    except Exception as e:
        print(str(e))
        raise
