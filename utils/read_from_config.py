# configparser
# config.read('example.ini')
# config.sections()


import configparser
import os


class Parameters:
    batch_size = None

    def __init__(self):
        self.batch_size = 128

    def __call__(self, *args, **kwargs):
        print(self.batch_size)


# Parser
def parser():
    parser = configparser.ConfigParser()
    path = ''
    print(parser.read('drishtypy/config.ini'))
    print(parser.sections())


parser()

param = Parameters()
# param()