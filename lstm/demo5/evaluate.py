import numpy as np
from data_generator import generate_sine_wave
import matplotlib.pyplot as plt

# 添加以下代码（放在import之后，函数定义之前）
plt.rcParams['font.sans-serif'] = ['PingFang']  # Mac系统
# 如果是Windows系统，使用：
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def prepare_test_input(sequence_length, amplitude, noise_scale=0.25):
    # Generate sine wave data
    _, data = generate_sine_wave(sequence_length + 50, amplitude, noise_scale)
    # Take the last sequence_length points as test input
    test_input = data[-sequence_length:]
    return test_input.reshape((1, sequence_length, 1))

def evaluate_models(simple_model, X, y, test_input):
    simple_prediction = simple_model.predict(test_input, verbose=0)
    
    print("\nModel Comparison:")
    print(f'Simple model predicted value: {simple_prediction[0][0]}')

    plt.figure(figsize=(20000, 6))  # 增加图表宽度
    
    # 绘制测试输入序列
    test_input_flat = test_input.reshape(-1)
    time_input = np.arange(len(test_input_flat))
    plt.plot(time_input, test_input_flat, label='测试输入序列', linewidth=1)
    
    # 绘制预测点
    plt.scatter(len(test_input_flat), simple_prediction[0][0], color='red', s=100, label='预测值')
    
    plt.title('预测结果可视化')
    plt.xlabel('时间步')
    plt.ylabel('振幅')
    plt.grid(True)
    plt.legend()
    
    # 调整x轴显示范围，使波形更加舒展
    current_xlim = plt.gca().get_xlim()
    plt.xlim(current_xlim[0], current_xlim[1])
    
    # 调整布局，防止标签被切掉
    plt.tight_layout()
    plt.show()
    
    simple_mse = simple_model.evaluate(X, y, verbose=0)
    
    print(f'\nSimple model MSE: {simple_mse}')
