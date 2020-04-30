import argparse
import configparser

from helper.shapenet.shapenetMapper import desc_to_id
from helper.shapenet.voxels.voxelGenerator import generateVoxels


parser = argparse.ArgumentParser(description="Generate voxels " +
                                 "from raw mesh data")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

cat_id = desc_to_id(config['data']['category'])

generateVoxels(config.get('pathConfiguration', 'inputPath'),
               config.get('pathConfiguration', 'outputPath'),
               config.get('binVoxel', 'executablePath'),
               cat_id, config.get('pathConfiguration', 'tempPath'),
               config.getint('voxel', 'lrange'),
               config.getint('voxel', 'urange'),
               config.getint('voxel', 'voxelDimensions'),
               config.getboolean('voxel', 'exact'),
               config.getboolean('voxel', 'dc'),
               config.getboolean('voxel', 'aw'),
               config.getboolean('voxel', 'reverse'),
               config.getboolean('voxel', 'overwrite'),
               config.getboolean('voxel', 'pb'))
