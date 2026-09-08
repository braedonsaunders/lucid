import unittest
import struct
import zlib
from fetch_reds_subset import unpack_entry

class RedsEntryTest(unittest.TestCase):
    def test_checked_identity_size_and_crc(self):
        name=b'train/train_sharp/001/00000000.png';data=b'\x89PNG\r\n\x1a\ncontent'
        crc=zlib.crc32(data)
        header=struct.pack('<IHHHHHIIIHH',0x04034b50,20,0,0,0,0,crc,len(data),len(data),len(name),0)
        row={'name':name.decode(),'compression':0,'size':len(data),'compressed_size':len(data),'crc':crc}
        self.assertEqual(unpack_entry(row,header,name+data),data)
        for bad in [{**row,'crc':0},{**row,'name':'wrong'},{**row,'size':99}]:
            with self.assertRaises(ValueError):unpack_entry(bad,header,name+data)
        with self.assertRaises(ValueError):unpack_entry(row,b'bad',name+data)


class GroupedRedsTest(unittest.TestCase):
    def test_grouped_entry_checks_and_resume(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from fetch_reds_subset import group_ranges, fetch_group
        data=b'\x89PNG\r\n\x1a\ncontent'
        rows=[]; archive=bytearray()
        for i in range(3):
            name=f'train/train_sharp/001/{i:08d}.png'.encode()
            crc=zlib.crc32(data)
            extra=b'local-extra'
            header=struct.pack('<IHHHHHIIIHH',0x04034b50,20,0,0,0,0,crc,len(data),len(data),len(name),len(extra))
            rows.append({'name':name.decode(),'compression':0,'size':len(data),'compressed_size':len(data),'crc':crc,'offset':len(archive)})
            archive.extend(header+name+extra+data)
        groups=group_ranges(rows)
        self.assertEqual(len(groups),1)
        archive.extend(b'\0'*65535)
        with tempfile.TemporaryDirectory() as temp, patch('fetch_reds_subset.read_range', side_effect=lambda start,n:bytes(archive[start:start+n])) as read:
            out=Path(temp)
            first=fetch_group(groups[0],out)
            self.assertEqual(read.call_count,1)
            second=fetch_group(groups[0],out)
            self.assertEqual(read.call_count,1)
            self.assertEqual(first,second)
            self.assertEqual(len(first),3)
            (out/first[0]['file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'existing frame changed'):
                fetch_group(groups[0],out)

    def test_group_bounds_and_bad_crc(self):
        from fetch_reds_subset import group_ranges
        rows=[{'name':'001/00000000.png','offset':0,'compressed_size':100000},
              {'name':'001/00000001.png','offset':200000,'compressed_size':100000}]
        groups=group_ranges(rows,maximum=200000)
        self.assertEqual(len(groups),2)
        self.assertTrue(all(end-start<=200000 for start,end,_ in groups))
        with self.assertRaises(ValueError):group_ranges(rows,maximum=1000)

if __name__=='__main__':unittest.main()
