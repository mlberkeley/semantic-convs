import unittest
from easydict import EasyDict

from scae.args import parse_args
from util.vis import print_edict


class TestCase(unittest.TestCase):
    # https://pypi.org/project/jsonargparse/

    def test_blank(self):
        p = parse_args(['--cfg', 'scae/config/default.yaml'])
        print_edict(p)
        # print(parser.format_help())
        self.assertEqual(True, False)

    def test_mnist(self):
        p = parse_args(['--cfg', 'scae/config/mnist.yaml'])
        print_edict(p)


if __name__ == '__main__':
    unittest.main()
