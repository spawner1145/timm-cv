"""
通过fulldanbooru2024数据集输出标签csv文件
"""
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import sys
import os

parquet_file_path = 'metadata.parquet'
csv_output_file = 'selected_tags.csv'

TAG_COLUMN_TO_CATEGORY = {
    'tag_string_general': 0,
    'tag_string_character': 4,
    'tag_string_copyright': 3,
    'tag_string_artist': 1,
    'tag_string_meta': 5,
}
RATING_TO_NAME = {
    'g': 'general',
    's': 'sensitive',
    'q': 'questionable',
    'e': 'explicit',
}
RATING_CATEGORY = 9

BASE_PROCESSING_COLUMNS = list(TAG_COLUMN_TO_CATEGORY.keys()) + ['rating']
COLUMNS_TO_READ_ALWAYS = BASE_PROCESSING_COLUMNS + ['id']
aggregated_data = {}

def process_row_data(row_series, current_aggregated_data):
    """处理单行数据 (Pandas Series)，并更新聚合数据字典。"""
    for col_name, category_id in TAG_COLUMN_TO_CATEGORY.items():
        if col_name in row_series and pd.notna(row_series[col_name]):
            tags_str = str(row_series[col_name]).strip()
            if tags_str:
                tags = tags_str.split(' ')
                for tag in tags:
                    tag = tag.strip()
                    if tag:
                        key = (tag, category_id)
                        current_aggregated_data[key] = current_aggregated_data.get(key, 0) + 1

    if 'rating' in row_series and pd.notna(row_series['rating']):
        rating_char = str(row_series['rating']).lower()
        rating_name = RATING_TO_NAME.get(rating_char)
        if rating_name:
            key = (rating_name, RATING_CATEGORY)
            current_aggregated_data[key] = current_aggregated_data.get(key, 0) + 1

def save_aggregated_data_to_csv(current_aggregated_data, file_path):
    """将聚合数据转换为DataFrame并保存到CSV文件。"""
    if not current_aggregated_data:
        print("尚无聚合数据可保存。")
        return

    output_list = [{'name': name, 'category': category, 'count': count}
                   for (name, category), count in current_aggregated_data.items()]
    df_output = pd.DataFrame(output_list)
    df_output.sort_values(by=['category', 'name', 'count'], ascending=[True, True, False], inplace=True)
    df_output.to_csv(file_path, index=False, encoding='utf-8')
    print(f"数据已成功写入 '{file_path}'")

def process_all_rows(parquet_file_obj, current_aggregated_data):
    """处理 Parquet 文件中的所有行。"""
    print("正在处理文件中的所有行...")
    
    rows_processed_count = 0
    num_row_groups = parquet_file_obj.num_row_groups

    if num_row_groups == 0 and parquet_file_obj.metadata.num_rows > 0:
        print("  文件没有明确的行组但包含数据，尝试一次性读取...")
        try:
            full_table = parquet_file_obj.read(columns=COLUMNS_TO_READ_ALWAYS)
            df_full_chunk = full_table.to_pandas(ignore_metadata=True, self_destruct=True)
            if not df_full_chunk.empty:
                for _, row_series in df_full_chunk.iterrows():
                    process_row_data(row_series, current_aggregated_data)
                    rows_processed_count += 1
            if rows_processed_count > 0:
                print(f"成功处理了文件中的全部 {rows_processed_count} 行。")
                return True
            else:
                print("文件中没有找到可处理的行。")
                return False
        except Exception as e:
            print(f"  尝试一次性读取所有行失败: {e}")
            return False


    for i in range(num_row_groups):
        print(f"  正在读取并处理行组 {i+1}/{num_row_groups}...")
        rg_table = parquet_file_obj.read_row_group(i, columns=COLUMNS_TO_READ_ALWAYS)
        df_rg_chunk = rg_table.to_pandas(ignore_metadata=True, self_destruct=True)

        if df_rg_chunk.empty:
            continue

        for _, row_series in df_rg_chunk.iterrows():
            process_row_data(row_series, current_aggregated_data)
            rows_processed_count += 1
            
    if rows_processed_count > 0:
        print(f"成功处理了文件中的全部 {rows_processed_count} 行。")
        return True
    else:
        print("文件中没有找到可处理的行 (可能文件本身为空或所有行组均为空)。")
        return False

def process_rows_by_id_column_exact_match(id_to_find_str, parquet_file_obj, current_aggregated_data):
    """根据 'id' 列的值进行精确 (字符串) 匹配并处理行。"""
    print(f"正在查找并处理 'id' 列精确匹配 '{id_to_find_str}' 的所有行...")
    
    rows_found_and_processed_count = 0
    num_row_groups = parquet_file_obj.num_row_groups

    for i in range(num_row_groups):
        rg_table = parquet_file_obj.read_row_group(i, columns=COLUMNS_TO_READ_ALWAYS)
        df_rg_chunk = rg_table.to_pandas(ignore_metadata=True, self_destruct=True)

        if df_rg_chunk.empty or 'id' not in df_rg_chunk.columns:
            continue
        
        matched_rows = df_rg_chunk[df_rg_chunk['id'].astype(str) == str(id_to_find_str)]

        if not matched_rows.empty:
            for _, row_series in matched_rows.iterrows():
                process_row_data(row_series, current_aggregated_data)
                rows_found_and_processed_count += 1
    
    if rows_found_and_processed_count > 0:
        print(f"成功找到并处理了 {rows_found_and_processed_count} 行，其 'id' 列值为 '{id_to_find_str}'。")
        return True
    else:
        print(f"未能找到 'id' 列值为 '{id_to_find_str}' 的行。")
        return False

def process_rows_by_id_column_numeric_range(start_val, end_val, parquet_file_obj, current_aggregated_data):
    """根据 'id' 列的数字范围筛选并处理行。"""
    print(f"正在查找并处理 'id' 列值在数字范围 [{start_val} - {end_val}] 内的所有行...")

    rows_found_and_processed_count = 0
    num_row_groups = parquet_file_obj.num_row_groups

    for i in range(num_row_groups):
        rg_table = parquet_file_obj.read_row_group(i, columns=COLUMNS_TO_READ_ALWAYS)
        df_rg_chunk = rg_table.to_pandas(ignore_metadata=True, self_destruct=True)

        if df_rg_chunk.empty or 'id' not in df_rg_chunk.columns:
            continue
        
        df_rg_chunk['numeric_id'] = pd.to_numeric(df_rg_chunk['id'], errors='coerce')
        df_rg_chunk.dropna(subset=['numeric_id'], inplace=True)

        if df_rg_chunk.empty:
            continue
            
        matched_rows = df_rg_chunk[
            (df_rg_chunk['numeric_id'] >= float(start_val)) & 
            (df_rg_chunk['numeric_id'] <= float(end_val))
        ]

        if not matched_rows.empty:
            for _, row_series in matched_rows.iterrows():
                process_row_data(row_series, current_aggregated_data)
                rows_found_and_processed_count += 1
                
    if rows_found_and_processed_count > 0:
        print(f"成功找到并处理了 {rows_found_and_processed_count} 行，其 'id' 列的数字值在范围 [{start_val} - {end_val}] 内。")
        return True
    else:
        print(f"未能找到 'id' 列的数字值在范围 [{start_val} - {end_val}] 内的行。")
        return False

def main():
    global aggregated_data

    print(f"--- 标签聚合脚本 (按 'id' 列值或 'all' 处理) ---")
    print(f"读取Parquet文件: {parquet_file_path}")
    print(f"将写入CSV文件: {csv_output_file}")
    print("-" * 30)

    if not os.path.exists(parquet_file_path):
        print(f"错误：Parquet文件 '{parquet_file_path}' 未找到！")
        sys.exit(1)

    try:
        parquet_file = pq.ParquetFile(parquet_file_path)
        total_rows_in_file = parquet_file.metadata.num_rows
        schema_names = parquet_file.schema.names
        print(f"成功打开Parquet文件。")
        print(f"文件总行数: {total_rows_in_file}, 总行组数: {parquet_file.num_row_groups}")
        if 'id' not in schema_names:
            print(f"关键错误：Parquet文件中未找到名为 'id' 的列。脚本无法按 'id' 值进行筛选。'all' 命令仍可工作。")
        if total_rows_in_file == 0:
            print("警告：Parquet文件为空。")
    except Exception as e:
        print(f"错误：无法打开或读取Parquet文件 '{parquet_file_path}' 的元数据: {e}")
        sys.exit(1)


    print("\n--- 交互式处理 ---")
    print("  输入 'all' 来处理文件中的所有行。")
    print("  输入 'id' 列的值范围 (例如 '1-100') 来按数字范围处理 'id' 列。")
    print("  输入单个 'id' 值 (例如 'item_abc' 或 '101') 来精确匹配 'id' 列的值。")
    print("  输入 'quit', 'exit' 或直接按 Enter 键结束。")

    while True:
        try:
            user_input = input(f"\n操作指令 ('all', 'id'范围如 '1-100', 单个'id'如 'item_x', 或 'quit'): ").strip()

            if not user_input or user_input.lower() in ['quit', 'exit']:
                print("收到退出指令，正在退出交互式处理。")
                break
            
            processed_successfully_in_iteration = False
            
            if user_input.lower() == 'all':
                if process_all_rows(parquet_file, aggregated_data):
                    processed_successfully_in_iteration = True
            else:
                parts = user_input.split('-', 1)
                is_range_query = False
                if len(parts) == 2:
                    try:
                        start_id_str, end_id_str = parts[0].strip(), parts[1].strip()
                        start_val = float(start_id_str)
                        end_val = float(end_id_str)
                        
                        if start_val > end_val:
                            print("ID范围错误：起始值不能大于结束值。")
                            continue
                        
                        if 'id' not in parquet_file.schema.names:
                            print("错误: 文件中没有 'id' 列, 无法按ID范围处理。")
                            continue
                        is_range_query = True
                        if process_rows_by_id_column_numeric_range(start_val, end_val, parquet_file, aggregated_data):
                            processed_successfully_in_iteration = True
                    except ValueError:
                        print(f"无法将 '{user_input}' 解析为数字范围，将尝试作为单个ID值进行精确匹配...")
                        pass 

                if not is_range_query:
                    if 'id' not in parquet_file.schema.names:
                        print(f"错误: 文件中没有 'id' 列, 无法按ID '{user_input}' 处理。")
                        continue
                    if process_rows_by_id_column_exact_match(user_input, parquet_file, aggregated_data):
                        processed_successfully_in_iteration = True
            
            if processed_successfully_in_iteration:
                 save_aggregated_data_to_csv(aggregated_data, csv_output_file)

        except ValueError as ve:
            print(f"输入错误或数值转换错误: {ve}")
        except Exception as e:
            print(f"处理过程中发生意外错误: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- 结束处理 ---")
    if aggregated_data:
        save_aggregated_data_to_csv(aggregated_data, csv_output_file)
    else:
        print("在本次运行中没有聚合任何数据。")
        if not os.path.exists(csv_output_file):
             pd.DataFrame(columns=['name', 'category', 'count']).to_csv(csv_output_file, index=False, encoding='utf-8')
             print(f"已创建空的 '{csv_output_file}' 文件并写入表头。")
    print("\n脚本执行完毕。")


if __name__ == "__main__":
    if not os.path.exists(parquet_file_path):
        print(f"'{parquet_file_path}' 未找到。正在创建一个包含多样化 'id' 列的虚拟文件用于测试...")
        
        ids_for_dummy = []
        ids_for_dummy.extend(list(range(1, 6)))
        ids_for_dummy.extend([str(i) for i in range(6, 11)])
        ids_for_dummy.extend([f'item_{chr(65+i)}' for i in range(3)])
        ids_for_dummy.extend([10, 10.5, '11', 'item_A_dup']) 
        ids_for_dummy.extend(list(range(20,23)))
        
        data_dict = {
            'id': ids_for_dummy,
            'tag_string_general': (["tagA tagB", "tagB tagC", "tagD", None, "tagA"] * 4),
            'tag_string_character': (["charA", "charB", None, "charC", "charA"] * 4),
            'tag_string_copyright': (["copyA", None, "copyB", "copyC", "copyA"] * 4),
            'tag_string_artist': (["artistA", "artistB", "artistC", None, "artistA"] * 4),
            'tag_string_meta': (["metaA", "metaB", "metaC", "metaD", None] * 4),
            'rating': (['g', 's', 'q', 'e', 'g'] * 4)
        }
        df_dummy = pd.DataFrame(data_dict)
        df_dummy['id'] = df_dummy['id'].astype(object)

        for col in TAG_COLUMN_TO_CATEGORY.keys():
            df_dummy[col] = df_dummy[col].astype(pd.StringDtype())
        df_dummy['rating'] = df_dummy['rating'].astype(pd.StringDtype())

        table = pa.Table.from_pandas(df_dummy, preserve_index=False)
        pq.write_table(table, parquet_file_path, row_group_size=6)
        print(f"虚拟文件 '{parquet_file_path}' 已创建，包含 {len(df_dummy)} 行。")
        print(f"示例 'id' 值 (注意混合类型): {ids_for_dummy[:5]}... {ids_for_dummy[5:10]}... {ids_for_dummy[10:13]}")
        print(f"测试建议: 'all', 范围如 '1-7', '8-10.5'. 单个ID如 'item_A', '4', '10.5'.")

    main()
