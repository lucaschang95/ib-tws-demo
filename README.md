# ibkr-tws

## macOS 创建虚拟环境
### 创建项目目录
mkdir my_project
cd my_project

### 创建虚拟环境
python -m venv .venv

### 激活虚拟环境（Windows）
.venv\Scripts\activate.bat

### 安装包
pip install yfinance

### 导出依赖列表
pip freeze > requirements.txt

### 退出虚拟环境
deactivate

## 文档
- [性能追踪](docs/PERFORMANCE.md) - 模型性能改进记录
- [Roadmap](docs/ROADMAP.md) - 项目规划和优先级