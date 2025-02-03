import numpy as np

# 加载 .npz 文件
npz_file_path = 'D:/En_eye_sore/user_1/exp1/prophesee/event_merge.npz'
data = np.load(npz_file_path)

# 访问名为 't' 的数组
x_array = data['x']
y_array = data['y']
t_array = data['t']
p_array = data['p']
# 定义偏移量和范围
offset =380001017


#range_min = 3.654599912e+08
# 8.11763609e+08
# 8.13778451e+08
range_min=3.95115677e+08
#range_max = 3.654799244e+08

range_max=3.97131292e+08

# 对数组中的每个元素加上偏移量
t_array_offset = t_array + offset


# 筛选出符合范围条件的元素
valid_indices = (t_array_offset >= range_min) & (t_array_offset <= range_max)
x_array=x_array[valid_indices]
y_array=y_array[valid_indices]
p_array=p_array[valid_indices]
valid_events = t_array_offset[valid_indices]



# 保存筛选后的数组为新的 .npz 文件
output_npz_file_path = '750_850.npz'
np.savez(output_npz_file_path, t=valid_events, p=p_array, x=x_array, y=y_array)

print(f"已将符合条件的 {len(valid_events)} 个事件保存到 {output_npz_file_path}")