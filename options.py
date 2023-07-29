import argparse
import json

class Options():
    def __init__(self, parser):
        self.parser = parser

    def initialize(self, parser):
        """Define the common options"""
        # ============== JSON =====================
        parser.add_argument('--load_json', action='store_false')
        parser.add_argument('--json_path', type=str, default='./configs/mouse_adult_brain.json')
        # ============== Additional =====================
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])

        return parser

    def gather_options(self):
        parser = self.initialize(self.parser)

        # Get json config
        opt, _ = parser.parse_known_args()
        if opt.load_json:
            with open(opt.json_path, "r") as f:
                js = f.read()
            config = json.loads(js)
            config.update(vars(parser.parse_known_args()[0]))
            opt = argparse.Namespace(**config)
        return opt

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()