from io import StringIO

from scipy.io import arff

content = """@relation A

@attribute color {ª,blue}

@data
ª"""

f = StringIO(content)
data, meta = arff.loadarff(f)