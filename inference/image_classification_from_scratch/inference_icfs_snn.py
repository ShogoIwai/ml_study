from argparse import ArgumentParser
import inference_icfs_com as iic

global opts
opts = {}

def parseOptions():
    argparser = ArgumentParser()
    argparser.add_argument('--mdl', help=':specify model file name') # use action='store_true' as flag
    argparser.add_argument('--img', help=':specify image dir path') # use action='store_true' as flag
    argparser.add_argument('--int', help=':int input flag', action='store_true') # use action='store_true' as flag
    args = argparser.parse_args()
    if args.mdl: opts.update({'mdl':args.mdl})
    if args.img: opts.update({'img':args.img})
    if args.int: opts.update({'int':args.int})

ext = r'\.jpg$'
image_size = (150, 150)

if __name__ == '__main__':
    parseOptions()
    if ('mdl' in opts.keys() and 'img' in opts.keys()):
        mdlfile = opts['mdl']
        imgdir = opts['img']

        if 'int' in opts.keys():
            iic.classification(mdlfile, imgdir, ext, image_size, int_flag=True, div=True)
        else:
            iic.classification(mdlfile, imgdir, ext, image_size, int_flag=False, div=True)
