import torch
import clip

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    # model, preprocess = clip.load("ViT-B/32", device=torch.cpu)
    model = clip.model.CLIP(
        512, 224, 12, 768, 32, 77, 49408, 512, 8, 12
    )
    try:
        torch.export.export(model, (x,))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
