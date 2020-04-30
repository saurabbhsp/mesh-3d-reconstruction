import os
import zipfile


class DataReader(object):
    base_path = None

    """check if the inputPath exist"""
    def __init__(self, base_path):
        if os.path.exists(base_path):
            self.base_path = base_path
        else:
            raise IOError("")

    """return files with cat_id"""
    def get_zip_path(self, cat_id):
        return os.path.join(self.base_path,
                            '%s.zip' % cat_id)

    def get_zip_file(self, cat_id):
        return zipfile.ZipFile(self.get_zip_path(cat_id))

    """returns data corresponding to the cat_id"""
    def list_archived_data(self, cat_id):

        archive_file = self.get_zip_file(cat_id)
        start = len(cat_id) + 1
        end = -len('model.obj')-1
        data = [
                n[start:end] for n in archive_file.namelist()
                if n[-4:] == '.obj']
        return data
