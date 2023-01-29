import zipfile
from struct import unpack


class ROIFrame:
    def __init__(self, bites: bytes):
        self.bites = bites
        assert bites[:4] == b'Iout'
        self.file_version, self.roi_type = unpack('>hh', bites[4:8])
        assert self.roi_type == 0, 'type is not polygon'
        self.top, self.left, self.bottom, self.right = unpack('>hhhh', bites[8:16])
        self.n_coordinates = unpack('>h', bites[16:18])[0]
        self.points = list()
        for i in range(self.n_coordinates):
            x = unpack('>h', bites[64+2*i:64+2*i+2])[0] + self.left
            y = unpack('>h', bites[64+2*i+2*self.n_coordinates:64+2*i+2*self.n_coordinates+2])[0] + self.top
            self.points.append((x, y))


class ROIFile:
    def __init__(self, filepath: str):
        zipped = zipfile.ZipFile(filepath)
        files = dict()
        for fn in zipped.filelist:
            files[fn.filename] = ROIFrame(zipped.read(fn.filename))


def test():
    rf = ROIFile('/home/hiaoxui/images/Ax2 mry-C2B-iai_AcquisitionBlock3.czi/RoiSet.zip')


if __name__ == '__main__':
    test()
