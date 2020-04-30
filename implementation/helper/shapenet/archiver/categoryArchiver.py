import os
from helper.shapenet.datawriter.writer import DataWriter


def create_archive(source_path, destination_path, cat_id, partial_archive):
    os.makedirs(destination_path, exist_ok=True)
    data_writer = DataWriter(destination_path)
    data_writer.archive_category(cat_id, source_path, partial_archive)


def merge_partial_archives(source_path, destination_path,
                           temp_path, cat_id):
    os.makedirs(destination_path, exist_ok=True)
    os.makedirs(temp_path, exist_ok=True)
    data_writer = DataWriter(destination_path)
    data_writer.extend_archive(cat_id, source_path, destination_path,
                               temp_path)
