import argparse
import configparser

from helper.shapenet.shapenetMapper import desc_to_id
from graphicUtils.pointCloud.generatePointCloudData import generatePointCloud


parser = argparse.ArgumentParser(description="Generate point cloud " +
                                 "from mesh data")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

cat_id = desc_to_id(config['data']['category'])

generatePointCloud(cat_id, config.get("pathConfiguration", "inputPath"),
                   config.get("pathConfiguration", "outputPath"),
                   config.get("cloud", "mode"),
                   config.getint("cloud", "pointCount"))
