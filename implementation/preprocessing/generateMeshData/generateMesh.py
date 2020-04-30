import argparse
import configparser

from helper.shapenet.shapenetMapper import desc_to_id
from graphicUtils.mesh.meshGenerator import generateMeshData

parser = argparse.ArgumentParser(description="Generate mesh data")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

cat_id = desc_to_id(config['data']['category'])

generateMeshData(cat_id, config.get("pathConfiguration", "inputPath"),
                 config.get("pathConfiguration", "outputPath"))
