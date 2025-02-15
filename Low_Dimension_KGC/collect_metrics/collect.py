import os
import pandas as pd

def extract_metrics_from_logs(base_path):
    metrics_data = []

    # 遍历base_path下的所有文件夹
    for folder in os.listdir(base_path):
        # 检查文件夹名称是否符合条件
        if folder.startswith("FB15k-237") or folder.startswith("WN18RR"):
            log_path = os.path.join(base_path, folder, "train.log")
            
            # 检查log文件是否存在
            if os.path.exists(log_path):
                with open(log_path, 'r') as file:
                    lines = file.readlines()
                
                # 定义一个字典来存储指标
                metrics = {}
                
                # 从后向前搜索以找到最后一次记录的模型指标
                for line in reversed(lines):
                    if "MRR at" in line:
                        metrics["MRR"] = float(line.split("MRR at")[1].split(":")[1].strip())
                    elif "MR at" in line:
                        metrics["MR"] = float(line.split("MR at")[1].split(":")[1].strip())
                    elif "HITS@1 at" in line:
                        metrics["HITS@1"] = float(line.split("HITS@1 at")[1].split(":")[1].strip())
                    elif "HITS@3 at" in line:
                        metrics["HITS@3"] = float(line.split("HITS@3 at")[1].split(":")[1].strip())
                    elif "HITS@10 at" in line:
                        metrics["HITS@10"] = float(line.split("HITS@10 at")[1].split(":")[1].strip())

                    # 检查是否已经获取所有需要的指标
                    if len(metrics) == 5:
                        metrics["Folder"] = folder
                        # 提取排序数字
                        folder_num = int(folder.rsplit('_', 1)[1])
                        metrics["Folder_num"] = folder_num
                        metrics_data.append(metrics)
                        break

    # 将数据写入DataFrame
    df = pd.DataFrame(metrics_data)
    # 按照Folder_num排序
    df.sort_values(by="Folder_num", inplace=True)
    # 删除辅助排序列
    df.drop(columns=["Folder_num"], inplace=True)
    # 重新排序列
    df = df[['Folder', 'MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']]
    
    # 将数据写入CSV文件
    df.to_csv(os.path.join(base_path, "metrics.csv"), index=False)
    return df


# 使用函数
base_path = "../models"  # 修改为实际的路径
df = extract_metrics_from_logs(base_path)
print(df)
