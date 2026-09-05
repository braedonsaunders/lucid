import tempfile, unittest
from pathlib import Path
from reference_pairs import resolve_reference
class ReferenceTests(unittest.TestCase):
    def test_unregistered_clip_cannot_silently_use_bbb(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'arbitrary.mp4';p.touch()
            with self.assertRaises(ValueError): resolve_reference(p)
            with self.assertRaises(ValueError): resolve_reference(p,p)
            reference=Path(directory)/'clean.mp4';reference.touch()
            self.assertEqual(resolve_reference(p,reference),str(reference))
    def test_registered_pair_requires_real_reference_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'crowdrun-360p-350k.mp4';p.touch()
            with self.assertRaises(ValueError):resolve_reference(p)
            reference=Path(directory)/'crowdrun-1080p.mp4';reference.touch()
            self.assertEqual(resolve_reference(p),str(reference))
if __name__=='__main__':unittest.main()
