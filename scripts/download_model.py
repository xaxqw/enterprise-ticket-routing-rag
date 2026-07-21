"""
    下载 LoRA 微调用的基座模型：Qwen2.5-0.5B-Instruct（从国内魔搭 ModelScope）
    
    运行：
    pip install -r requirements-finetune.txt
    python scripts/download_model.py
    
    说明：Embedding / 重排模型现在都走硅基流动在线API，无需本地下载。
    """
import os

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "base")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    try:
        from modelscope import snapshot_download
    except ImportError:
        print(" 未安装 modelscope，请先执行：")
        print(" pip install -r requirements-finetune.txt -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return

        print("开始下载 Qwen2.5-0.5B-Instruct（CPU 可微调的小模型）...")
        model_dir = snapshot_download(
            "Qwen/Qwen2.5-0.5B-Instruct",
            cache_dir=BASE_DIR,
            )
        print(f"\n 模型下载完成，缓存路径：{model_dir}")

        # ModelScope 把模型放在 ./models/base/models/Qwen--xxx/snapshots/<rev>，
        # 但 run_finetune.py / .env(LLM_LOCAL_MODEL) 期望标准路径 ./models/base/Qwen/Qwen2.5-0.5B-Instruct。
        # 这里把模型同步到标准路径，保证下载脚本与训练/推理脚本口径一致（幂等，已存在则跳过）。
        import shutil
        canonical = os.path.join(BASE_DIR, "Qwen", "Qwen2.5-0.5B-Instruct")
        if os.path.isfile(os.path.join(canonical, "config.json")):
            print(f" 标准路径已就绪：{canonical}")
        else:
            print(f"⏳ 同步模型到标准路径：{canonical} ...")
            if os.path.isdir(canonical):
                shutil.rmtree(canonical)
                os.makedirs(os.path.dirname(canonical), exist_ok=True)
                shutil.copytree(model_dir, canonical)
                print(f" 已同步到标准路径：{canonical}")

                print("接下来运行： python scripts/run_finetune.py")


                if __name__ == "__main__":
                    main()
