import pkg_resources

__version__ = pkg_resources.get_distribution("yews").version

from yews import data
from yews import models
from yews import train
from yews import cpic
