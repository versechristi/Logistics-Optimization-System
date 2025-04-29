# data/solomon_parser.py
# -*- coding: utf-8 -*-
import re

def load_solomon_data(filename):
    """
    加载 Solomon VRPTW benchmark 数据集。

    Args:
        filename (str): Solomon 数据集文件的路径。

    Returns:
        dict: 包含解析后的数据的字典，包括：
            'depot': (x, y) 坐标
            'customers': [(x, y)] 客户坐标列表
            'demands': [demand] 客户需求列表
            # 可以根据需要添加其他信息，如服务时间窗口等
    """
    data = {
        'depot': None,
        'customers': [],
        'demands': []
    }

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()

        # 查找节点坐标和需求信息的起始行
        node_coord_start = -1
        demand_start = -1
        for i, line in enumerate(lines):
            if line.strip() == 'NODE_COORD_SECTION':
                node_coord_start = i + 1
            elif line.strip() == 'DEMAND_SECTION':
                demand_start = i + 1
                break

        if node_coord_start != -1 and demand_start != -1:
            # 解析节点坐标
            for i in range(node_coord_start, demand_start - 1):
                parts = lines[i].strip().split()
                if len(parts) == 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    if node_id == 1:
                        data['depot'] = (x, y)
                    else:
                        data['customers'].append((x, y))

            # 解析需求
            customer_index = 0
            for i in range(demand_start, len(lines)):
                parts = lines[i].strip().split()
                if len(parts) == 2:
                    node_id = int(parts[0])
                    demand = int(parts[1])
                    if node_id > 1:
                        if customer_index < len(data['customers']):
                            data['demands'].append(demand)
                            customer_index += 1
                        else:
                            print(f"警告: 需求信息比客户坐标多，忽略节点ID {node_id} 的需求。")
                elif line.strip() == 'DEPOT_SECTION': # 部分 Solomon 文件可能在此处结束坐标和需求信息
                    break
                elif line.strip() == 'TIME_WINDOW_SECTION': # 如果有时间窗信息，可以继续解析
                    pass # 留待后续实现
                elif line.strip() == 'SERVICE_TIME_SECTION': # 如果有服务时间信息，可以继续解析
                    pass # 留待后续实现

            # 确保解析到的需求数量与客户数量一致
            if len(data['demands']) != len(data['customers']):
                print(f"警告: 解析到的需求数量 ({len(data['demands'])}) 与客户数量 ({len(data['customers'])}) 不一致。")
                # 可以根据具体情况处理，例如截断或填充

        else:
            print(f"错误: 文件 {filename} 格式不正确，无法找到 NODE_COORD_SECTION 或 DEMAND_SECTION。")
            return None

    except FileNotFoundError:
        print(f"错误: 文件 {filename} 未找到。")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

    return data

if __name__ == '__main__':
    # 示例用法 (你需要将 'your_solomon_instance.txt' 替换为实际的文件名)
    solomon_file = 'data/examples/c101.txt'  # 假设在 data/examples 目录下有一个 Solomon 实例文件
    solomon_data = load_solomon_data(solomon_file)

    if solomon_data:
        print("成功加载 Solomon 数据:")
        print("仓库:", solomon_data['depot'])
        print("前 5 个客户坐标:", solomon_data['customers'][:5])
        print("前 5 个客户需求:", solomon_data['demands'][:5])
        print(f"总客户数: {len(solomon_data['customers'])}")
        print(f"总需求数: {len(solomon_data['demands'])}")