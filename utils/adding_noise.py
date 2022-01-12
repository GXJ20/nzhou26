import starfile
import pathlib
import sys
import os
star_paths = list(pathlib.Path(sys.argv[1]).glob('*.star'))
#star_paths_sort = sorted(star_paths, key=lambda star: len(starfile.read(star)), reverse=True)
def count():
    star_path_empty = []
    for item in star_paths:
        try:
            if len(starfile.read(item)) < 180:
                print(item.name)
                print(len(starfile.read(item)))
                os.system(f'scp -v {item} {sys.argv[2]}')
                star_path_empty.append(item)
        except:
            print(item)
    print(len(star_path_empty))
def erase_normal():
    os.system('mkdir -p total_messy')
    for item in star_paths:
        meta = starfile.read(item)
        meta = meta.dropna()
        starfile.write(meta, f'total_messy/{item.name}')
        print(f'Wrote {item.name}')

erase_normal()