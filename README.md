# 物流路径优化系统 (Logistics Optimization System)

## 项目简介

项目是一个多站点、二级车辆路径问题（MD-2E-VRPSD）的物流优化系统，包含无人机协同和拆分配送功能。系统通过图形用户界面（GUI）进行交互，允许用户配置问题参数、选择优化算法、运行优化过程并可视化结果。该问题场景涉及：一个或多个物流中心为多个销售网点补货，销售网点再使用车辆和无人机共同为客户配送货物。

项目采用 Python 语言开发，提供了一个基于 Tkinter 的图形用户界面 (GUI)，用于配置参数、生成模拟数据、运行优化算法以及可视化分析结果。

当前实现的优化算法包括：

* 遗传算法 (Genetic Algorithm, GA)
* 模拟退火 (Simulated Annealing, SA)
* 粒子群优化 (Particle Swarm Optimization, PSO)
* 贪心启发式算法 (Greedy Heuristic)

## 项目结构

logistics_optimization/
│
├── main.py                  # 主程序入口，负责初始化环境并启动 GUI
├── requirements.txt         # 项目依赖库列表 (如果存在)
├── README.md                # 项目说明文件

├── config/                  # 配置文件目录
│   └── default_config.ini   # 数据生成、载具属性、目标函数权重、算法参数的默认配置

├── core/                    # 核心业务逻辑与计算模块
│   ├── __init__.py          # Core 包的初始化文件，暴露核心组件 (如 haversine, cost_function, problem_utils 中的 SolutionCandidate/Stage 2 生成器, route_optimizer)
│   ├── cost_function.py     # 成本与时间计算（目标函数），负责评估解决方案的总成本、总时间(Makespan)和未满足需求，并计算加权目标值，处理两级、无人机、分割配送。现在接受 Stage 2 启发式和距离函数作为参数。
│   ├── distance_calculator.py # 地理距离计算 (Haversine 公式)，基础工具。
│   ├── problem_utils.py     # 核心问题通用工具模块。包含 Stage 2 配送趟次生成启发式、SolutionCandidate 类、初始解/邻域生成函数、排列变异操作符等被多个模块（特别是 core 和 algorithm）共享的、与具体算法无关但与问题本身相关的工具。
│   └── route_optimizer.py   # 优化流程编排，负责数据加载、预处理、调用选定的算法、收集并整合所有算法的结果，并触发可视化和报告生成。

├── data/                    # 数据生成与加载模块
│   ├── __init__.py          # Data 包的初始化文件 (如果需要暴露数据处理函数)
│   ├── data_generator.py    # 生成合成的地理位置和客户需求数据。
│   └── solomon_parser.py    # 解析 Solomon VRPTW 标准数据集 (需要适配 MD-2E-VRPSD)。

├── algorithm/               # 优化算法实现模块
│   ├── __init__.py          # Algorithm 包的初始化文件，暴露各算法的主运行函数 (如 run_genetic_algorithm 等)。
│   ├── greedy_heuristic.py  # 贪心启发式算法实现。通过简单的贪婪规则构建解决方案，并使用核心评估函数和 Stage 2 启发式。
│   ├── genetic_algorithm.py # 遗传算法 (GA) 实现。优化 Stage 1 路线，使用 SolutionCandidate 类，并依赖核心评估函数和 Stage 2 启发式。
│   ├── simulated_annealing.py # 模拟退火 (SA) 算法实现。基于局部搜索和概率接受，使用 SolutionCandidate 类和核心通用工具中的邻域生成函数及核心评估函数。
│   ├── pso_optimizer.py     # 粒子群优化 (PSO) 算法实现。优化 Stage 1 路线，使用排列式 PSO 变体，自定义 Particle 类（继承 SolutionCandidate），并依赖核心评估函数。
│   └── utils.py             # Algorithm 通用工具模块，此文件可能只包含少量或不含任何内容，除非有严格意义上只属于算法层面的通用辅助函数。

└── gui/                     # 图形用户界面模块
    ├── __init__.py          # GUI 包的初始化文件 (如果需要暴露 GUI 组件或函数)
    ├── main_window.py       # GUI 主窗口类，处理用户交互、参数设置、算法执行调用、结果显示（地图、图表、报告）。
    └── utils.py             # GUI 控件创建和布局的辅助函数。

└── visualization/           # 结果可视化模块
    ├── __init__.py          # Visualization 包的初始化文件 (如果需要暴露可视化函数)
    ├── map_generator.py     # 生成交互式地图 (Folium)，可视化路线和点位。
    └── plot_generator.py    # 生成 Matplotlib 图表，可视化算法迭代曲线和结果对比。

## 功能特点

* **参数化数据生成**：可配置生成随机的物流中心、销售网点、客户位置及客户需求量。
* **灵活配置**：支持自定义车辆和无人机的数量、载重、成本、速度、无人机续航距离等参数。
* **多算法支持**：实现了遗传算法、模拟退火、粒子群优化和贪心启发式算法，方便对比分析。
* **图形用户界面 (GUI)**：提供友好的交互界面，便于参数设置、运行控制和结果查看。
* **丰富的结果可视化**：
    * **静态路径图**：使用 Matplotlib 绘制优化后的两级路径概览图。
    * **交互式地图**：使用 Folium 生成带图例的 HTML 地图，可缩放、平移、查看具体点信息和路径。
    * **迭代收敛图**：展示启发式算法在迭代过程中成本（或其他目标值）的变化情况。
    * **结果对比图**：使用条形图直观对比不同算法最终的综合成本和总时间。
* **详细报告**：生成包含路径细节、成本、时间、服务客户数等信息的文本报告。
* **配置持久化**：通过 `config/default_config.ini` 文件加载和保存参数配置。
* **结果导出**：支持将生成的地图、图表和报告保存到本地文件。

## 安装指南

1.  **克隆仓库**:
    ```bash
    git clone <your-repository-url>
    cd logistics_optimization
    ```
2.  **(推荐)** **创建并激活虚拟环境**:
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```
3.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    *(请确保 `requirements.txt` 文件包含了如 `folium`, `matplotlib`, `numpy` 等所有必需的库)*

## 使用方法

1.  **启动程序**:
    在项目根目录下运行：
    ```bash
    python main.py
    ```
    这将启动图形用户界面。

2.  **GUI 操作流程**:
    * **参数配置 (左侧面板)**:
        * 在 "Data Generation" 标签页设置数据生成参数（中心点、网点、客户数量、半径、需求范围等）。
        * 在 "Vehicle Params" 和 "Drone Params" 标签页设置车辆和无人机属性。
        * 在 "Objective Func" 标签页调整成本和时间在目标函数中的权重，以及未满足需求的惩罚值。
        * 在下方的算法参数 Notebook 中（GA, SA, PSO 标签页）配置所选算法的特定参数。
    * **生成数据**: 点击 "Generate Data Points" 按钮生成模拟数据，并在右侧 "Route Map" 标签页预览点位分布。
    * **选择算法**: 在 "Algorithm Selection & Execution" 区域的多选框中选择一个或多个需要运行和对比的算法。
    * **运行优化**: 点击 "Run Optimization" 按钮。程序将在后台执行所选算法。
    * **查看结果 (右侧面板)**:
        * **Route Map (Plot)**: 查看由 Matplotlib 生成的静态路径图。可以通过下拉菜单切换不同算法的结果图。
        * **Iteration Curve**: 查看各算法在优化过程中的成本收敛曲线图。
        * **Results Comparison**: 查看各算法最终优化结果（成本、时间）的条形对比图。
        * **Detailed Report**: 查看包含路径详情、各项成本/时间指标的文本报告。
        * **(交互式地图)** 点击 "Open Interactive Map" 按钮可在浏览器中打开对应算法的 Folium 交互式地图 (前提是 Folium 已正确安装且地图已成功生成到 `output/maps/` 目录下)。
    * **保存结果**: 使用各结果标签页下方的 "Save Plot" 或 "Save Report" 按钮，可以将图表或报告保存到 `output/` 目录下的相应子文件夹中。
    * **保存参数**: 点击 "Save Parameters" 按钮可将当前 GUI 界面上的所有参数写回 `config/default_config.ini` 文件。

## 主要依赖库

* **Tkinter** (Python 标准库): 用于构建图形用户界面。
* **Matplotlib**: 用于绘制静态路径图、迭代曲线图和对比图。
* **Folium**: 用于生成交互式 HTML 地图。
* **NumPy**: 用于进行数值计算，特别是在数据生成和某些算法中。

*(本项目由Mr VerseChristi编写完成,若有疑问或者需要提供数据可以联系versechristi@gmail.com)*
