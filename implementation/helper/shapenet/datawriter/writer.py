import os
import zipfile
import uuid


def _write_zip(source_path, destination_path):
    with zipfile.ZipFile(destination_path, 'a', allowZip64=True) as zip:
        for dir_name, sub_dirs, files in os.walk(source_path):

            for file in files:
                relative_path = os.path.relpath(dir_name, source_path)
                zip.write(os.path.join(dir_name, file),
                          os.path.join(relative_path, file))


class DataWriter(object):
    base_path = None

    def __init__(self, base_path):
        self.base_path = base_path

    def get_zip_path(self, cat_id):
        return os.path.join(self.base_path,
                            '%s.zip' % cat_id)

    def archive_category(self, cat_id, source_path, partial_archive):
        zip_path = None
        if not partial_archive:
            zip_path = self.get_zip_path(cat_id)
        else:
            unique_id = uuid.uuid4()
            zip_path = self.get_zip_path('%s_%s' % (cat_id, unique_id))

        source_path_category = os.path.join(source_path, cat_id)
        _write_zip(source_path_category, zip_path)

    def extend_archive(self, cat_id, source_path, destination_path, temp_path):
        partial_archives = [x for x in os.listdir(source_path)
                            if x.startswith(str(cat_id)+"_")
                            and x.endswith(".zip")]
        for partial_archive in partial_archives:
            zip = zipfile.ZipFile(os.path.join(source_path, partial_archive))
            zip.extractall(temp_path)
        zip_path = self.get_zip_path(cat_id)
        _write_zip(temp_path, zip_path)
