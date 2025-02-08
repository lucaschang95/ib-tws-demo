import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_sine_wave, create_sequences, prepare_data

def visualize_time_series(time, data):
    """
    仅可视化时间序列数据
    
    Args:
        time: 时间序列
        data: 原始数据
    """
    plt.figure(figsize=(20000, 6))  # 增加图表宽度
    plt.plot(time, data, linewidth=1)
    plt.xlim(0, 300)
    plt.title('时间序列数据')
    plt.xlabel('时间')
    plt.ylabel('振幅')
    plt.grid(True)
    
    # 调整x轴显示范围，使波形更加舒展
    current_xlim = plt.gca().get_xlim()
    plt.xlim(current_xlim[0], current_xlim[1])
    
    # 调整布局，防止标签被切掉
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成数据
    time, wave_data = generate_sine_wave(n_points=3000, amplitude=10)
    
    # 可视化
    visualize_time_series(time, wave_data) 