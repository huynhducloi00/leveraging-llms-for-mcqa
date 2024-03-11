import os


def idx_to_ltr(idx):
    return chr(idx + ord("A"))


def ltr_to_idx(ltr):
    return ord(ltr) - ord("A")


def make_dir_if_does_not_exist(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

