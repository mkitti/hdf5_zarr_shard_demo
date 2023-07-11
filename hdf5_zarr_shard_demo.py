"""
    hdf5_zarr_shard_demo.py

Demonstrate a HDF5 File that is also a ZEP0002 Zarr Shard with checksum

Mark Kittisopikul, Ph.D.
Software Engineer II
Computational Methods
Scientific Computing
Janelia Research Campus
"""

import struct
import jenkins_cffi
import h5py

def write_simple_chunked(filename):
    """
    Create a HDF5 file with 16 chunks. Importantly, the compressed chunks
    require 4 bytes to describe their size.
    """
    with h5py.File(filename, "w", libver="v112") as f:
        c = 2048
        h5ds = f.create_dataset(
            "zarrshard",
            (4*c,4*c),
            dtype="uint8",
            chunks=(c,c),
            compression="gzip"
        ) 
        h5ds[  0:c,     0:c  ] = 1
        h5ds[  0:c,     c:c*2] = 2
        h5ds[  0:c,   c*2:c*3] = 3
        h5ds[  0:c,   c*3:c*4] = 4
        h5ds[  c:c*2,   0:c  ] = 5
        h5ds[  c:c*2,   c:c*2] = 6
        h5ds[  c:c*2, c*2:c*3] = 7
        h5ds[  c:c*2, c*3:c*4] = 8
        h5ds[c*2:c*3,   0:c  ] = 9
        h5ds[c*2:c*3,   c:c*2] = 10
        h5ds[c*2:c*3, c*2:c*3] = 11
        h5ds[c*2:c*3, c*3:c*4] = 12
        h5ds[c*3:c*4,   0:c  ] = 13
        h5ds[c*3:c*4,   c:c*2] = 14
        h5ds[c*3:c*4, c*2:c*3] = 15
        h5ds[c*3:c*4, c*3:c*4] = 16

def verify_checksum(b):
    """
    Apply Jenkins lookup3 hash and compare to the last four bytes
    """
    hash_bytes = jenkins_cffi.hashlittle(bytes(b[:-4])).to_bytes(4, "little")
    return b[-4:] == hash_bytes

def set_checksum(b):
    """
    Apply Jenkins lookup3 hash and set the last four bytes
    """
    hash_bytes = jenkins_cffi.hashlittle(bytes(b[:-4])).to_bytes(4, "little")
    ba = bytearray(b)
    ba[-4:] = hash_bytes
    return bytes(ba)

def verify_superblock(filename):
    """
    Demonstrate superblock checksum verification
    """
    with open(filename, "rb") as f:
        superblock_bytes = f.read(48)
    return verify_checksum(superblock_bytes)

def read_fixed_array_metadata(filename):
    """
    Read the HDF5 file extracting the
    1) superblock
    2) Fixed Array Header (FAHD)
    3) Fixed Array Data Block (FADB)
    """
    with open(filename, "rb") as f:
        b = f.read()

        superblock_bytes = b[:48]

        fahd_position = b.find(b"FAHD")
        f.seek(fahd_position)
        fahd_bytes = f.read(28)

        n_chunks = int.from_bytes(fahd_bytes[8:16], "little")
        entry_size = fahd_bytes[6]
        
        fadb_position = b.find(b"FADB")
        fadb_bytes = f.read(18 + n_chunks*entry_size)
    return superblock_bytes, fahd_position, fahd_bytes, fadb_position, fadb_bytes

def move_fadb_to_end(filename):
    """
    Move the Fixed Address Data Block (FADB) to the end of the file
    """
    superblock_bytes, fahd_position, fahd_bytes, fadb_position, fadb_bytes = read_fixed_array_metadata(filename)
    with open(filename, "rb+") as f:
        # Seek to end and write FADB
        # TODO: Expand FADB to 4 byte chunk sizes if less than 4 bytes
        # TODO: Throw an error if FADB chunk size is larger than 4 bytes
        f.seek(0, 2)
        new_fadb_position = f.tell()
        f.write(fadb_bytes)
        eof = f.tell()

        # Rewrite superblock eof
        f.seek(0)
        superblock_bytes = bytearray(superblock_bytes)
        superblock_bytes[-20:-12] = eof.to_bytes(8, "little")
        superblock_bytes = set_checksum(superblock_bytes)
        f.write(superblock_bytes)

        # Adjust FAHD and overwrite
        f.seek(fahd_position)
        fahd_bytes = bytearray(fahd_bytes)
        fahd_bytes[-12:-4] = new_fadb_position.to_bytes(8, "little")
        fahd_bytes = set_checksum(fahd_bytes)
        f.write(fahd_bytes)
        
        # Zero out old FADB
        f.seek(fadb_position)
        f.write(bytes(len(fadb_bytes)))

def read_zarr_shard_chunk_metadata(filename):
    with open(filename, "rb") as f:
        f.seek(-2*8*16-4, 2)
        b = f.read(2*8*16)
        offset_nbyte_pairs = [x for x in struct.iter_unpack("<qq", b)]
    return offset_nbyte_pairs


def demo(filename = "hdf5_zarr_shard_demo.h5"):
    write_simple_chunked("original_" + filename)
    write_simple_chunked(filename)
    move_fadb_to_end(filename)
    with h5py.File(filename, "r") as f:
        print("Data via h5py:")
        print(f["zarrshard"][:])
    print("")
    print("Zarr Shard Chunk Offset Nbyte Pairs:")
    print(read_zarr_shard_chunk_metadata(filename))

if __name__ == "__main__":
    demo()
