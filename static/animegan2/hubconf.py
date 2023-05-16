import torch


def generator(pretrained=True, device="cpu", progress=True, check_hash=True):
    from model import Generator

    known = {
        name: f"static/animegan2/weights/{name}.pt"
        for name in [
            'celeba_distill', 'face_paint_512_v1', 'face_paint_512_v2', 'paprika'
        ]
    }

    device = torch.device(device)
    model = Generator().to(device)

    if type(pretrained) == str:
        # Look if a known name is passed, otherwise assume it's a URL
        ckpt_url = known.get(pretrained, pretrained)
        pretrained = True
    else:
        ckpt_url = known.get('face_paint_512_v2')

    if pretrained is True:
        state_dict = torch.load(
            ckpt_url,
        )
        model.load_state_dict(state_dict)

    return model


def face2paint(device="cpu", size=512, side_by_side=False):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image

    # 函数的第一行是函数定义。它接受五个参数：model 表示需要使用的PyTorch模型；img表示待处理的PIL.Image对象；size表示输出图像的大小（默认为512）；side_by_side
    # 表示是否将原始图像与输出图像并排显示（默认为False）；device表示模型所在的设备（默认是CPU）。
    #
    # 第二行首先获取输入图像的尺寸 w 和 h。然后计算出输入图像的最短边长 s，以便在进行裁剪和缩放操作时，将图像裁剪为正方形并调整到指定大小。
    #
    # 在第三行中，使用 crop() 函数对图像进行裁剪，保留中心位置的正方形区域。由于图像大小是可变的，因此实际的裁剪边界也会不同。这里使用了 Python 语言内置的整除运算符 // 来计算边界。
    #
    # 第四行使用 resize() 函数将裁剪后的正方形图像调整为指定大小，并使用 Image.LANCZOS 参数指定插值算法，以获得更清晰的图像。
    #
    # 接下来，在第五行中，to_tensor() 函数逐像素地将图像转换为PyTorch tensor，然后将其乘以2并减去1，以实现归一化（即将像素值缩放到 [-1, 1] 的范围内）。
    #
    # 第六行将输入数据送入 PyTorch 模型进行计算。这里使用了 model() 函数对输入数据进行前向传递，得到输出 tensor。由于计算过程中不需要反向传播，因此使用 torch.no_grad()
    # 上下文管理器可以节省内存和计算时间。
    #
    # 在第七行中，使用 cpu() 函数将输出 tensor 从 GPU 中移动到 CPU 上，并使用索引 [0] 取出 batch 中的第一个元素。由于该模型是对单张图像进行计算，因此输出 shape 为 (1, C, H,
    # W)，其中 C 表示通道数，H 和 W 分别表示输出图像的高度和宽度。所以使用 [0] 索引操作来取出 C × H × W 大小的 tensor。
    #
    # 在第八行中，如果 side_by_side 参数为 True，则使用 torch.cat() 函数将原始图像和生成的油画风格图像沿着 W 轴（即宽度方向）拼接起来，以便并排显示。torch.cat() 函数将两个
    # tensor 拼接成一个新的 tensor，其 shape 为 (1, C, H, 2W)。否则，直接返回生成的油画风格图像。
    #
    # 最后，在第九行中，将输出 tensor 转换为 PIL.Image 对象，并将其返回，即得到最终的结果。由于输出 tensor 的取值范围在 [-1, 1] 之间，因此使用乘以0.5再加上0.5的方式将像素值缩放为 [0,
    # 1] 的范围内。同时还需要使用 clip() 函数防止像素值溢出范围。
    def face2paint(
            model: torch.nn.Module,
            img: Image.Image,
            size: int = size,
            side_by_side: bool = side_by_side,
            device: str = device,
    ) -> Image.Image:
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)

        with torch.no_grad():
            input = to_tensor(img).unsqueeze(0) * 2 - 1
            output = model(input.to(device)).cpu()[0]

            if side_by_side:
                output = torch.cat([input[0], output], dim=2)

            output = (output * 0.5 + 0.5).clip(0, 1)

        return to_pil_image(output)

    return face2paint
