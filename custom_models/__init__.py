from .Perceptron import Perceptron
from .GBC import GBC
from .KNN import KNN
from .LSVC import LSVC
from .SVC import SVC

MODELS = {
    'GBC': GBC(),
    'Perceptron': Perceptron(),
    'KNN': KNN(),
    'SVC': SVC(),
    'LSVC': LSVC()
}

