import unittest
from unittest.mock import patch
from io import BytesIO
from pathlib import Path
import tempfile
from fetch_sources import y4m_layout, download_ranges


class Y4MLayoutTests(unittest.TestCase):
    def test_parallel_ranges_reconstruct_exact_bytes_and_reject_ignored_ranges(self):
        data = bytes(range(256)) * 65537
        def response(request, timeout):
            first, last = map(int, request.headers['Range'].removeprefix('bytes=').split('-'))
            result = BytesIO(data[first:last+1])
            result.status = 206
            result.headers = {'Content-Range': f'bytes {first}-{last}/{len(data)}'}
            return result
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'excerpt'
            with patch('fetch_sources.urllib.request.urlopen', side_effect=response):
                download_ranges('https://example.invalid/raw', target, len(data), 2)
            self.assertEqual(target.read_bytes(), data)
            def ignored(request, timeout):
                result = response(request, timeout)
                result.status = 200
                return result
            with patch('fetch_sources.urllib.request.urlopen', side_effect=ignored):
                with self.assertRaisesRegex(ValueError, 'honor'):
                    download_ranges('https://example.invalid/raw', target, 16, 2)

    def test_omitted_chroma_uses_standard_eight_bit_420_default(self):
        header = b'YUV4MPEG2 W1920 H1080 F50:1 Ip A1:1\n'
        self.assertEqual(y4m_layout(header), (1920, 1080, 3110400))
        for chroma in (b'420jpeg', b'420mpeg2', b'420paldv'):
            self.assertEqual(y4m_layout(header[:-1] + b' C' + chroma + b'\n'), (1920, 1080, 3110400))

    def test_eight_bit_422_is_sized_for_conversion(self):
        # derf 1080p live-action masters are 4:2:2; the excerpt is preserved as 4:2:0.
        self.assertEqual(y4m_layout(b'YUV4MPEG2 W1920 H1080 F30000:1001 Ip A1:1 C422\n'), (1920, 1080, 4147200))

    def test_high_bit_depth_and_other_byte_layouts_are_rejected(self):
        for chroma in (b'420p10', b'444', b'mono'):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                y4m_layout(b'YUV4MPEG2 W1920 H1080 C' + chroma + b'\n')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            y4m_layout(b'YUV4MPEG2 W1920 H1080 XYSCSS=420P10\n')


if __name__ == '__main__':
    unittest.main()
