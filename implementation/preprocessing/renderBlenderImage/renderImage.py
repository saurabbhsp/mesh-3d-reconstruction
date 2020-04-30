import argparse
import configparser

from helper.shapenet.shapenetMapper import desc_to_id
from helper.shapenet.blender.render import render_cat

parser = argparse.ArgumentParser(description="Render images using blender")
parser.add_argument('configFile', help='Config file path')
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.configFile)

cat_id = desc_to_id(config['data']['category'])

render_cat(config.get('data', 'inputPath'), config.get('data', 'outputPath'),
           config.get('data', 'tmpExtraction'),
           config.get('externalScript', 'pythonBlenderScript'),
           cat_id, config.getint('image', 'imageCount'),
           config.getint('image', 'shapeX'), config.getint('image', 'shapeY'),
           config.getboolean('image', 'overwrite'), config.getboolean('image',
           'reverse'), config.getboolean('image', 'fixedMeshes'),
           config.get('blender', 'blenderPath'),
           config.getint('image', 'lrange'), config.getint('image', 'urange'))
