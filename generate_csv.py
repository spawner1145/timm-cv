"""
测试用生成selected_tags.csv的脚本,扫描数据集文件夹所有文件生成
"""
import os
import glob
import pandas as pd
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_selected_tags(input_folder, output_csv_path="selected_tags.csv", txt_tag_separator=' '):
    """
    扫描输入文件夹中的 .txt 文件，统计标签频率，并生成 selected_tags.csv

    Args:
        input_folder (str): 包含 .txt 标签文件的文件夹路径
                            脚本会查找与图片同名的 .txt 文件 (不直接处理图片)
        output_csv_path (str): 生成的 selected_tags.csv 文件的保存路径
        txt_tag_separator (str): .txt 文件中标签之间的分隔符
    """
    logging.info(f"开始扫描文件夹: {input_folder}")
    
    all_tags = []
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))

    if not txt_files:
        logging.warning(f"在文件夹 {input_folder} 中没有找到 .txt 文件")
        empty_df = pd.DataFrame(columns=['name', 'count'])
        empty_df.to_csv(output_csv_path, index=False)
        logging.info(f"已生成空的 {output_csv_path}")
        return

    logging.info(f"找到 {len(txt_files)} 个 .txt 文件，开始读取标签...")

    for txt_file_path in txt_files:
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                tags_str = f.read().strip()
                if tags_str:
                    tags_in_file = [tag.strip() for tag in tags_str.split(txt_tag_separator) if tag.strip()]
                    all_tags.extend(tags_in_file)
        except Exception as e:
            logging.error(f"读取或处理文件 {txt_file_path} 时出错: {e}")

    if not all_tags:
        logging.warning("未从任何 .txt 文件中提取到标签")
        empty_df = pd.DataFrame(columns=['name', 'count'])
        empty_df.to_csv(output_csv_path, index=False)
        logging.info(f"已生成空的 {output_csv_path} (因为未提取到标签)")
        return

    logging.info(f"总共提取到 {len(all_tags)} 个标签实例 (包含重复)")

    tag_counts = Counter(all_tags)
    logging.info(f"统计得到 {len(tag_counts)} 个唯一标签")

    tags_df = pd.DataFrame(tag_counts.items(), columns=['name', 'count'])
    tags_df = tags_df.sort_values(by='count', ascending=False).reset_index(drop=True)

    try:
        tags_df.to_csv(output_csv_path, index=False)
        logging.info(f"selected_tags.csv 已成功生成并保存到: {output_csv_path}")
        logging.info(f"CSV 文件包含 {len(tags_df)} 行")
    except Exception as e:
        logging.error(f"保存 CSV 文件到 {output_csv_path} 时出错: {e}")

if __name__ == "__main__":
    input_folder = "data/images"
    output_csv = "selected_tags.csv"
    separator = ","

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"已创建输出目录: {output_dir}")

    generate_selected_tags(input_folder, output_csv, separator)
