import sys
import argparse
from generate import mkimage, initialize

# check if python3 is used
if sys.version_info < (3, 0):
    print("This programs need python3x to run. Aborting.")
    sys.exit(1)

p = argparse.ArgumentParser(description='Generate images from background and elements.')
p.add_argument('-b', '--backgrounds'
               , help='directory containing background images')
p.add_argument('-c', '--classes'
               , help='directory containing element images, one subdirectory per class')
p.add_argument('-s', '--single', action='store_true'
               , help='generate images containing one class elements only')
p.add_argument('-n'
               , help='number of images to generate')
p.add_argument('-e'
               , help='max number of elements per image')
p.add_argument('-o'
               , help='directory to store generated images')
p.add_argument('-m'
               , help='directory to store generated masks')
p.add_argument('-i' , help='index of first sample')
args = p.parse_args()
print(args)

species = {    
    "Benthosema": ['Benthosema Glaciale', (70,103)],
    "BlueWhiting": ['Micromesistius Poutassou', (220,550)],
    "Herring": ['Clupea Harengus', (300,440)],
    "Mackerel": ['Scomber Scombrus', (301,601)]
}

objects, names, backgrounds = initialize(backgrounds_dir=args.backgrounds, classes_dir=args.classes, species_list=species)

n = int(args.n)
initial = int(args.i)
if args.e == None: e=6
else: e=int(args.e)

print(args.o, args.m)

for i in range(initial,n+initial+1):
    print('n = ' + str(int(n)) + ' current = ' + str(i), end='\r')
    mkimage('test_%d' % int(i), objects, names, backgrounds, species, maxobjs=e, output_dir=args.o, mask_dir=args.m,  single=args.single)
    
