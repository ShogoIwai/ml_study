from argparse import ArgumentParser
from google_images_download import googleimagesdownload
import re

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--tgt', help=':specify target want to get') # use action='store_true' as flag
    argparser.add_argument('--knd', help=':specify kind want to get') # use action='store_true' as flag
    argparser.add_argument('--lim', help=':optional, specify limit') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.knd: opts.update({'knd':args.knd})
    if args.tgt: opts.update({'tgt':args.tgt})
    if args.lim: opts.update({'lim':args.lim})

def main(limit='10'):
    outdir = opts['tgt']
    outdir = re.sub('\s', '_', outdir)
    imgdir = opts['knd']
    imgdir = re.sub('\s', '_', imgdir)
    arguments = {
        'keywords':f"{opts['tgt']}",
        'output_directory':f"{outdir}",
        'prefix_keywords':f"{opts['knd']}",
        'image_directory':f"{imgdir}",
        'limit':f"{limit}",
        'format':'jpg',
        'print_urls':True,
    } 

    print(arguments)
    response = googleimagesdownload()
    response.download(arguments)

if __name__ == '__main__':
    parseOptions()
    if ('tgt' in opts.keys() and 'knd' in opts.keys()):
        if ('lim' in opts.keys()):
            main(opts['lim'])
        else:
            main()
