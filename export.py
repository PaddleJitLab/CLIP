import torch
import clip

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    text_input = torch.randn(3, 77)
    model = clip.model.CLIP(
        512, 224, 12, 768, 32, 77, 49408, 512, 8, 12
    )
    try:
        torch.export.export(model, (x, text_input))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
