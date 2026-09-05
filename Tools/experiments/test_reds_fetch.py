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

if __name__=='__main__':unittest.main()
