import json
import os
import subprocess


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def modify_parameters(params, modifications):
    for key, value in modifications.items():
        if key in params:
            params[key] = value
    return params


def main():
    filepath = 'paras.json'  # 替换为你的JSON文件路径
    params = load_json(filepath)

    # 定义不同的参数集
    parameter_sets = [
        # #{'batch_size': 32, 'num_epochs': 20, 'lr': 0.0003}, #example
        # {'num_epochs': 20, 'lr': 0.0005},
        # {'num_epochs': 20,'embed_poi': 250, 'lr': 0.0005},  #后面波动大
        {'num_epochs': 36, 'lr': 0.0002},
        # {'num_epochs': 41,'lr':0.0001},
        # 添加更多参数集
    ]

    for i, param_set in enumerate(parameter_sets):
        # if i<6: continue

        new_params = modify_parameters(params.copy(), param_set)
        new_filepath = f'params_set_{i+1}.json'
        save_json(new_filepath, new_params)
        print("save?")
        # 在这里添加代码，用新参数集运行你的训练脚本
        # 运行训练脚本并等待其完成
        result = subprocess.run(
            ['python', 'train.py', '--config', new_filepath], capture_output=True, text=True)

        print(f"Standard Output:\n{result.stdout}")
        print(f"Standard Error:\n{result.stderr}")
        print(f"Return Code: {result.returncode}")

        # 打印训练脚本的输出结果
        print(f"Running training with parameters set {i+1}:")


if __name__ == '__main__':
    main()
