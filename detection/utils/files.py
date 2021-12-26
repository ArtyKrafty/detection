import os


def walk_to_level(path, level=None):
    if level is None:
        yield from os.walk(path)
        return

    path = path.rstrip(os.path.sep)
    num_sep = path.count(os.path.sep)
    for root, dirs, files in os.walk(path):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def list_files(path, valid_exts=None, level=None, contains=None):
    for (root_dir, dir_names, filenames) in walk_to_level(path, level):
        for filename in sorted(filenames):
            if contains is not None and contains not in filename:
                continue
            ext = filename[filename.rfind("."):].lower()
            if valid_exts and ext.endswith(valid_exts):
                file = os.path.join(root_dir, filename)
                yield file