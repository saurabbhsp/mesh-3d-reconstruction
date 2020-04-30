import argparse
import configparser
import os

from helper.shapenet.shapenetMapper import desc_to_id
from helper.shapenet.archiver.categoryArchiver import create_archive

parser = argparse.ArgumentParser(description="Archive rendered images")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

cat_id = desc_to_id(config['data']['category'])
create_archive(os.path.join(config.get("data", "inputPath"), 'images'),
               os.path.join(config.get("data", "outputPath"), 'images'),
               cat_id, config.getboolean("archive", "partialArchive"))
