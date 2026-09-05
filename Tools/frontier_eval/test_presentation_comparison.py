import unittest
from PIL import Image
from evaluate_sequences import present_4x_at_2x


class PresentationComparisonTests(unittest.TestCase):
    def test_adapter_cannot_hide_wrong_model_scale(self):
        with self.assertRaisesRegex(ValueError, 'genuine 4x'):
            present_4x_at_2x(Image.new('RGB', (64, 32)), (32, 16))
        result = present_4x_at_2x(Image.new('RGB', (128, 64), (42, 60, 90)), (32, 16))
        self.assertEqual(result.size, (64, 32))
        self.assertEqual(result.getpixel((30, 15)), (42, 60, 90))


if __name__ == '__main__':
    unittest.main()
