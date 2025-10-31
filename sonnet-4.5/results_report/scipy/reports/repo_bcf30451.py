from io import StringIO
from scipy.io import arff

content = r"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"0.0\n"
"""

f = StringIO(content)
data, meta = arff.loadarff(f)