# 使features目录成为一个Python包
# 这样可以让Python正确处理模块导入

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 