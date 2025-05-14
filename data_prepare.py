# -*- coding: utf-8 -*-
"""
Danbooru2024图片下载、.txt标签生成与CSV聚合脚本,总之就是准备数据集用的脚本
"""
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import os
import sys

try:
    from cheesechaser.datapool import Danbooru2024WebpDataPool
except ImportError:
    print("错误: cheesechaser 库未找到。")
    print("请确保已安装: pip install \"huggingface_hub[hf_xet]\" cheesechaser>=0.2.0")
    sys.exit(1)

try:
    import generate_csv as csv_aggregator
except ImportError:
    print("错误: generate_csv.py 未找到。")
    print("请确保 generate_csv.py 与本脚本在同一目录下。")
    sys.exit(1)
except Exception as e:
    print(f"导入 generate_csv.py 时发生其他错误: {e}")
    sys.exit(1)

PARQUET_FILE_PATH = 'metadata.parquet'
IMAGES_DST_DIR = 'data/images/'
CSV_OUTPUT_FILE = 'data/selected_tags.csv'
TAG_COLUMN_FOR_TXT_CAPTIONS = 'tag_string'
ID_COLUMN_NAME = 'id'

BASE_PROCESSING_COLUMNS_FOR_CSV = list(csv_aggregator.TAG_COLUMN_TO_CATEGORY.keys()) + ['rating']
DOWNLOADER_SPECIFIC_COLS_TO_READ = [ID_COLUMN_NAME, TAG_COLUMN_FOR_TXT_CAPTIONS]
ALL_REQUIRED_COLUMNS_IN_PARQUET = list(
    set(csv_aggregator.COLUMNS_TO_READ_ALWAYS + DOWNLOADER_SPECIFIC_COLS_TO_READ)
)
if ID_COLUMN_NAME not in ALL_REQUIRED_COLUMNS_IN_PARQUET:
    ALL_REQUIRED_COLUMNS_IN_PARQUET.append(ID_COLUMN_NAME)


def safe_to_int(value):
    if value is None: return None
    try: return int(float(value))
    except (ValueError, TypeError): return None


def fetch_data_and_prepare_for_processing(parquet_file_obj,
                                          command_type,
                                          command_criteria=None,
                                          save_csv_every_n_row_groups=float('inf'),
                                          csv_output_filepath_for_batch_save=None):
    downloader_data_list = []
    processed_int_ids_for_downloader_dedup = set()
    rows_passed_to_csv_aggregator_count = 0
    row_groups_processed_since_last_csv_save = 0
    num_row_groups = parquet_file_obj.num_row_groups

    def process_dataframe_chunk(df_chunk_to_process_param):
        nonlocal rows_passed_to_csv_aggregator_count
        df_chunk_to_process = df_chunk_to_process_param.copy()

        if ID_COLUMN_NAME not in df_chunk_to_process.columns: return
        if TAG_COLUMN_FOR_TXT_CAPTIONS not in df_chunk_to_process.columns:
            df_chunk_to_process.loc[:, TAG_COLUMN_FOR_TXT_CAPTIONS] = None
        
        missing_csv_cols_in_this_chunk = [
            col for col in csv_aggregator.BASE_PROCESSING_COLUMNS if col not in df_chunk_to_process.columns
        ]

        if command_type == 'all':
            filtered_rows_df = df_chunk_to_process
        elif command_type == 'exact_id':
            try:
                id_to_match_int = int(command_criteria)
                numeric_ids_col = pd.to_numeric(df_chunk_to_process[ID_COLUMN_NAME], errors='coerce')
                filtered_rows_df = df_chunk_to_process[numeric_ids_col == id_to_match_int]

            except ValueError:
                print(f"    输入的单个ID '{command_criteria}' 不是有效的整数。")
                filtered_rows_df = pd.DataFrame()
        elif command_type == 'range_id':
            start_val, end_val = command_criteria
            numeric_ids = pd.to_numeric(df_chunk_to_process[ID_COLUMN_NAME], errors='coerce')
            valid_mask = numeric_ids.notna()
            if not valid_mask.any(): filtered_rows_df = pd.DataFrame()
            else:
                filtered_rows_df = df_chunk_to_process[valid_mask &
                    (numeric_ids >= float(start_val)) & (numeric_ids <= float(end_val))
                ]
        else: filtered_rows_df = pd.DataFrame()

        for _, row_series in filtered_rows_df.iterrows():
            try:
                csv_aggregator.process_row_data(row_series, csv_aggregator.aggregated_data)
                rows_passed_to_csv_aggregator_count += 1
            except Exception as e_csv:
                current_id_for_error = row_series.get(ID_COLUMN_NAME, '未知ID')

            original_id_val = row_series.get(ID_COLUMN_NAME)
            int_id = safe_to_int(original_id_val) 
            if int_id is not None and int_id not in processed_int_ids_for_downloader_dedup:
                caption_tag_string = row_series.get(TAG_COLUMN_FOR_TXT_CAPTIONS)
                downloader_data_list.append({'id': int_id, 'caption_tags': caption_tag_string})
                processed_int_ids_for_downloader_dedup.add(int_id)

    if num_row_groups == 0 and parquet_file_obj.metadata.num_rows > 0:
        print("  Parquet文件没有明确的行组但包含数据，尝试一次性读取...")
        try:
            table = parquet_file_obj.read(columns=ALL_REQUIRED_COLUMNS_IN_PARQUET)
            df_chunk_single_read = table.to_pandas(ignore_metadata=True, self_destruct=True)
            if not df_chunk_single_read.empty:
                process_dataframe_chunk(df_chunk_single_read)
        except Exception as e: print(f"  一次性读取Parquet数据以进行处理时发生错误: {e}")
    else:
        for i in range(num_row_groups):
            if num_row_groups <= 20 or (i+1) % max(1, num_row_groups // 20) == 0 or i == num_row_groups -1 :
                 print(f"  正在扫描Parquet行组 {i+1}/{num_row_groups}...")
            try:
                rg_table = parquet_file_obj.read_row_group(i, columns=ALL_REQUIRED_COLUMNS_IN_PARQUET)
                df_chunk_from_rg = rg_table.to_pandas(ignore_metadata=True, self_destruct=True)
                if not df_chunk_from_rg.empty:
                    process_dataframe_chunk(df_chunk_from_rg)
            except Exception as e:
                print(f"    读取或处理行组 {i+1} 时发生错误: {e}。跳过此行组。")
                continue
            
            if csv_output_filepath_for_batch_save and num_row_groups > 0 :
                row_groups_processed_since_last_csv_save += 1
                if row_groups_processed_since_last_csv_save >= save_csv_every_n_row_groups:
                    if csv_aggregator.aggregated_data:
                        print(f"  (CSV分批保存: 已处理 {row_groups_processed_since_last_csv_save} 个行组，正在保存到 '{csv_output_filepath_for_batch_save}'...)")
                        try:
                            csv_aggregator.save_aggregated_data_to_csv(csv_aggregator.aggregated_data, csv_output_filepath_for_batch_save)
                            row_groups_processed_since_last_csv_save = 0
                        except Exception as e_batch_csv_save:
                            print(f"    CSV分批保存时出错: {e_batch_csv_save}")
    
    if rows_passed_to_csv_aggregator_count > 0:
        print(f"  (总计已将 {rows_passed_to_csv_aggregator_count} 行匹配数据传递给CSV聚合器)")
    return downloader_data_list


def download_images_and_write_captions(id_caption_list, img_dst_dir, pool_instance, max_dl_workers=12):
    if not id_caption_list:
        print("没有提供ID用于下载。")
        return

    image_ids_for_download = []
    valid_items_for_captions = []
    for item in id_caption_list:
        dl_id = item['id']
        if isinstance(dl_id, int):
             image_ids_for_download.append(dl_id)
             valid_items_for_captions.append(item)
        else:
            print(f"  警告: ID '{dl_id}' 不是预期的整数类型，跳过。")

    if not image_ids_for_download:
        print("经过筛选，没有有效的整数ID可供下载。")
        return

    print(f"准备下载 {len(image_ids_for_download)} 张图片...")
    os.makedirs(img_dst_dir, exist_ok=True)

    try:
        pool_instance.batch_download_to_directory(
            resource_ids=image_ids_for_download,
            dst_dir=img_dst_dir,
            max_workers=max_dl_workers,
        )
        print("批量图片下载请求已执行。")
    except Exception as e:
        print(f"批量下载过程中发生严重错误: {e}")
        print("将尝试为任何可能已成功下载的图片生成标签文件。")

    print("开始生成 .txt 标签文件...")
    txt_files_created_count = 0
    for item in valid_items_for_captions:
        current_int_id = item['id']
        raw_caption_tags = item['caption_tags']
        image_file_path = os.path.join(img_dst_dir, f"{current_int_id}.webp")
        txt_caption_path = os.path.join(img_dst_dir, f"{current_int_id}.txt")

        if os.path.exists(image_file_path):
            tags_for_file = ""
            if pd.notna(raw_caption_tags) and str(raw_caption_tags).strip():
                tags_for_file = str(raw_caption_tags).strip().replace(' ', ',')
            try:
                with open(txt_caption_path, 'w', encoding='utf-8') as f: f.write(tags_for_file)
                txt_files_created_count += 1
            except Exception as e: print(f"  错误: 无法为 ID {current_int_id} 写入标签文件 '{txt_caption_path}': {e}")
            
    print(f"成功为 {txt_files_created_count} 张已下载的图片生成了 .txt 标签文件。")


def main_downloader_and_processor():
    print(f"使用Parquet元数据: {PARQUET_FILE_PATH}")
    print(f"图片和.txt标签将保存到: {IMAGES_DST_DIR}")
    csv_output_dir_abs = os.path.abspath(os.path.dirname(CSV_OUTPUT_FILE) if os.path.dirname(CSV_OUTPUT_FILE) else ".")
    print(f"聚合后的CSV ('selected_tags.csv') 将保存到目录: {csv_output_dir_abs}")
    print(f".txt标签内容来自Parquet列: '{TAG_COLUMN_FOR_TXT_CAPTIONS}'")
    print(f"CSV聚合逻辑由导入的 'generate_csv.py' 提供。")
    print(f"注意: '{ID_COLUMN_NAME}'列中的值将被视为整数进行匹配和处理。")
    print("-" * 30)

    if not os.path.exists(PARQUET_FILE_PATH):
        print(f"错误：Parquet元数据文件 '{PARQUET_FILE_PATH}' 未找到！程序无法继续。")
        sys.exit(1)

    try:
        parquet_file = pq.ParquetFile(PARQUET_FILE_PATH)
        
        arrow_schema = parquet_file.schema_arrow 
        schema_names = arrow_schema.names

        if ID_COLUMN_NAME not in schema_names:
            print(f"关键错误：ID列 '{ID_COLUMN_NAME}' 在Parquet文件中不存在。脚本无法继续。")
            sys.exit(1)
        
        id_arrow_field = arrow_schema.field(ID_COLUMN_NAME)
        id_arrow_type_str = str(id_arrow_field.type)
        print(f"  Parquet中 '{ID_COLUMN_NAME}' 列的Arrow数据类型: {id_arrow_type_str}")
        if not (pa.types.is_integer(id_arrow_field.type) or \
                (pa.types.is_floating(id_arrow_field.type) and id_arrow_type_str.endswith('.0'))): # Check for float that are whole numbers
            print(f"  警告: '{ID_COLUMN_NAME}' 列的Arrow数据类型 ('{id_arrow_type_str}') 可能不完全是整数，但脚本会尝试按整数处理。")


        if TAG_COLUMN_FOR_TXT_CAPTIONS not in schema_names:
            print(f"警告：用于.txt文件的标签列 '{TAG_COLUMN_FOR_TXT_CAPTIONS}' 在Parquet文件中不存在。.txt文件内容将为空。")
        
        missing_csv_cols = [col for col in csv_aggregator.COLUMNS_TO_READ_ALWAYS if col not in schema_names]
        if missing_csv_cols:
            print(f"警告：CSV聚合所需的以下列在Parquet文件中缺失: {missing_csv_cols}。CSV结果可能不完整。")

        print(f"文件总行数: {parquet_file.metadata.num_rows}, 总行组数: {parquet_file.num_row_groups}")

    except Exception as e:
        print(f"错误：无法打开或读取Parquet文件 '{PARQUET_FILE_PATH}' 的元数据: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        pool = Danbooru2024WebpDataPool()
        print("Danbooru2024WebpDataPool 初始化成功。")
    except Exception as e:
        print(f"错误: 初始化 Danbooru2024WebpDataPool 失败: {e}")
        sys.exit(1)

    csv_dir_for_makedirs = os.path.dirname(CSV_OUTPUT_FILE)
    if csv_dir_for_makedirs and not os.path.exists(csv_dir_for_makedirs):
        os.makedirs(csv_dir_for_makedirs, exist_ok=True)

    print("\n--- 交互式指令输入 ---")
    print("  'all': 处理元数据中的所有ID。")
    print("  '范围' (例如 '1000-2000'): 处理指定数字范围内的ID。")
    print("  '单个ID' (例如 '123456'): 处理指定的单个整数ID。")
    print("  'quit' 或 'exit': 退出脚本。")

    while True:
        user_input = input("\n请输入操作指令: ").strip()
        if not user_input or user_input.lower() in ['quit', 'exit']:
            print("收到退出指令。")
            break
        
        command_type_for_fetch = None
        criteria_for_fetch = None
        apply_batch_csv_save_for_this_command = False
        batch_save_interval = 20

        if user_input.lower() == 'all':
            command_type_for_fetch = 'all'
            apply_batch_csv_save_for_this_command = True 
            total_rg = parquet_file.num_row_groups
            if total_rg > 10 : 
                 batch_save_interval = max(1, min(50, total_rg // 10)) 
            print(f"准备处理所有ID... (CSV将大约每 {batch_save_interval} 个行组保存一次)")
        else:
            parts = user_input.split('-', 1)
            is_range_query_success = False
            if len(parts) == 2:
                try:
                    start_val = int(parts[0].strip())
                    end_val = int(parts[1].strip())
                    if start_val > end_val:
                        print("ID范围错误：起始值不能大于结束值。请重新输入。")
                        continue
                    command_type_for_fetch = 'range_id'
                    criteria_for_fetch = (start_val, end_val)
                    is_range_query_success = True
                    print(f"准备处理ID范围 [{start_val}-{end_val}]...")
                except ValueError:
                    print(f"无法将 '{user_input}' 解析为整数范围，将尝试作为单个ID值处理...")
            
            if not is_range_query_success:
                try:
                    single_id_val_int = int(user_input)
                    command_type_for_fetch = 'exact_id'
                    criteria_for_fetch = single_id_val_int
                    print(f"准备处理单个ID值 '{single_id_val_int}'...")
                except ValueError:
                    print(f"输入 '{user_input}' 不是有效的 'all'、'整数-整数'范围或单个整数ID。请重试。")
                    continue

        # 1. 从Parquet获取数据，聚合CSV数据到内存，并收集下载信息
        data_for_downloader = fetch_data_and_prepare_for_processing(
            parquet_file, command_type_for_fetch, criteria_for_fetch,
            save_csv_every_n_row_groups=batch_save_interval if apply_batch_csv_save_for_this_command else float('inf'),
            csv_output_filepath_for_batch_save=CSV_OUTPUT_FILE if apply_batch_csv_save_for_this_command else None
        )
        
        # 2. 在开始下载之前，保存当前聚合的CSV数据
        if csv_aggregator.aggregated_data:
            print(f"准备在下载操作前保存/更新聚合后的CSV数据到 '{CSV_OUTPUT_FILE}'...")
            try:
                csv_aggregator.save_aggregated_data_to_csv(csv_aggregator.aggregated_data, CSV_OUTPUT_FILE)
                print(f"CSV数据已在下载前成功保存/更新。")
            except Exception as e_csv_save:
                print(f"错误: 在下载前保存CSV文件 '{CSV_OUTPUT_FILE}' 时发生错误: {e_csv_save}")
        elif not os.path.exists(CSV_OUTPUT_FILE):
            print(f"'{CSV_OUTPUT_FILE}' 不存在且当前无聚合数据，尝试创建空的CSV文件。")
            try:
                csv_output_dir_for_empty_file = os.path.dirname(CSV_OUTPUT_FILE)
                if csv_output_dir_for_empty_file and not os.path.exists(csv_output_dir_for_empty_file):
                     os.makedirs(csv_output_dir_for_empty_file, exist_ok=True)
                pd.DataFrame(columns=['name', 'category', 'count']).to_csv(CSV_OUTPUT_FILE, index=False, encoding='utf-8')
                print(f"已创建空的 '{CSV_OUTPUT_FILE}' 文件。")
            except Exception as e_empty_csv:
                print(f"错误: 尝试创建空的 '{CSV_OUTPUT_FILE}' 文件时出错: {e_empty_csv}")

        # 3. 如果有数据需要下载，则执行下载和.txt生成
        if data_for_downloader:
            print(f"从元数据中获取了 {len(data_for_downloader)} 个项目用于下载和.txt文件生成。")
            download_images_and_write_captions(data_for_downloader, IMAGES_DST_DIR, pool)
        else:
            print("未能从元数据中找到任何有效的ID进行处理。")

    if csv_aggregator.aggregated_data:
        print("正在执行最终的CSV数据保存...")
        try:
            csv_aggregator.save_aggregated_data_to_csv(csv_aggregator.aggregated_data, CSV_OUTPUT_FILE)
            print(f"最终CSV数据已成功保存到 '{CSV_OUTPUT_FILE}'。")
        except Exception as e_csv_save_final: print(f"最终保存CSV文件 '{CSV_OUTPUT_FILE}' 时出错: {e_csv_save_final}")
    else:
        print("在本次会话中，没有数据被聚合到CSV。")
        if not os.path.exists(CSV_OUTPUT_FILE):
            try:
                csv_output_dir_for_final_empty = os.path.dirname(CSV_OUTPUT_FILE)
                if csv_output_dir_for_final_empty and not os.path.exists(csv_output_dir_for_final_empty):
                     os.makedirs(csv_output_dir_for_final_empty, exist_ok=True)
                pd.DataFrame(columns=['name', 'category', 'count']).to_csv(CSV_OUTPUT_FILE, index=False, encoding='utf-8')
                print(f"已创建空的 '{CSV_OUTPUT_FILE}' 文件。")
            except Exception as e_empty_csv: print(f"尝试创建空的 '{CSV_OUTPUT_FILE}' 文件时出错: {e_empty_csv}")
            
    print("\n脚本执行完毕。")

if __name__ == '__main__':         
    main_downloader_and_processor()
