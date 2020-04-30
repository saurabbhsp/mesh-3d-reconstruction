import argparse
import configparser

from helper.shapenet.shapenetMapper import desc_to_id
from helper.shapenet.archiver.categoryArchiver import merge_partial_archives

parser = argparse.ArgumentParser(description="Merge partial archives")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

cat_id = desc_to_id(config['data']['category'])
merge_partial_archives(config.get("data", "inputPath"),
                       config.get("data", "outputPath"),
                       config.get("data", "tempPath"), cat_id,
                       )
