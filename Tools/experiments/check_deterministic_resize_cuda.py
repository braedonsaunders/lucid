"""One protected CUDA preflight: bicubic equivalence and bit-exact repeated gradients."""
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from deterministic_resize import bicubic


def main():
    torch.manual_seed(719)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    for source, target in [((17, 23), (28, 32)), ((32, 40), (19, 17)), ((176, 176), (224, 224))]:
        torch.use_deterministic_algorithms(False)
        original = torch.rand(2, 3, *source, device='cuda', requires_grad=True)
        expected = F.interpolate(original, target, mode='bicubic', align_corners=False, antialias=True)
        gradient = torch.randn_like(expected)
        expected.backward(gradient)
        torch.use_deterministic_algorithms(True)
        outputs, gradients = [], []
        for _ in range(2):
            image = original.detach().clone().requires_grad_()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                actual = bicubic(image, target)
            actual.backward(gradient)
            outputs.append(actual.detach()); gradients.append(image.grad.detach())
        forward_error = float((outputs[0]-expected.detach()).abs().max())
        backward_error = float((gradients[0]-original.grad).abs().max())
        assert forward_error < 1e-6 and backward_error < 3e-6, (forward_error, backward_error)
        assert torch.equal(outputs[0], outputs[1]) and torch.equal(gradients[0], gradients[1])
        rows.append({'source': source, 'target': target, 'max_forward_error': forward_error,
                     'max_backward_error': backward_error, 'repeat_output_and_gradient_exact': True})
    report = {'complete': True, 'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
              'rows': rows, 'code_sha256': hashlib.sha256(Path(__file__).with_name('deterministic_resize.py').read_bytes()).hexdigest()}
    Path('resize-cuda-preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
