import os
from time import strftime, localtime
from termcolor import colored


class IO():

    def __init__(self, log_dir=None):

        self.log_dir = log_dir

    def print_info(self, info_string, quite=False):

        info = '[{0}][INFO] {1}'.format(self.get_local_time(), info_string)
        print colored(info, 'green')

    def print_warning(self, warning_string):

        warning = '[{0}][WARNING] {1}'.format(self.get_local_time(), warning_string)

        print colored(warning, 'blue')

    def print_error(self, error_string):

        error = '[{0}][ERROR] {1}'.format(self.get_local_time(), error_string)

        print colored(error, 'red')

    def get_local_time(self):

        return strftime("%d %b %Y %Hh%Mm%Ss", localtime())

    def read_file_list(self, filelist):

        pfile = open(filelist)
        filenames = pfile.readlines()
        pfile.close()

        filenames = [f.strip() for f in filenames]

        return filenames

    def split_pair_names(self, filenames, base_dir):

        filenames = [c.split(' ') for c in filenames]
        filenames = [(os.path.join(base_dir, c[0]), os.path.join(base_dir, c[1])) for c in filenames]

        return filenames
