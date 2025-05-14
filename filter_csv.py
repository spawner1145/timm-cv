"""
这个脚本是用来平衡selected_tags.csv文件中的数据的,可以选择取数量最多的前n行,可以根据count数量筛选
可以选择只处理某些category,对于danbooru数据集输出的csv建议先处理一遍
"""
import pandas as pd

def filter_top_n_per_category(input_csv_path, output_csv_path, n=None, min_count_value=None, selected_categories=None):
    """
    读取CSV文件，对指定的category或所有category，按'count'降序排序。
    如果指定了n (正整数)，则保留每个category中count数量最多的n行。
    如果n未指定或为None/0，则保留所有符合条件的行。
    可以选择性地要求count列的值大于某个指定值，并筛选特定category，然后保存到新的CSV文件。

    参数:
    input_csv_path (str): 输入CSV文件的路径。
    output_csv_path (str): 输出CSV文件的路径。
    n (int, optional): 每个category保留的行数。
                       如果为正整数，则取最大的n行。
                       如果为None、0或负数，则保留该category下所有符合其他条件的行 (按count降序)。
                       默认为None。
    min_count_value (float or int, optional): count列必须大于的最小值。
                                             如果为None，则不进行此筛选。默认为None。
    selected_categories (list, optional): 一个包含要处理的category值的列表 (例如 [0, 1, 'A'])。
                                          如果为None或空列表，则处理所有category。默认为None。
    """
    try:
        df = pd.read_csv(input_csv_path)
        if 'category' not in df.columns or 'count' not in df.columns:
            print("错误：CSV文件中必须包含 'category' 和 'count' 列。")
            return

        try:
            df['count'] = pd.to_numeric(df['count'])
        except ValueError:
            print("错误：'count' 列包含无法转换为数值的值。请检查数据。")
            return
        filters_applied_messages = []
        if selected_categories is not None and len(selected_categories) > 0:
            original_row_count = len(df)
            df = df[df['category'].isin(selected_categories)]
            current_row_count = len(df)
            filters_applied_messages.append(f"category 为 {selected_categories} (筛选前行数: {original_row_count}, 筛选后行数: {current_row_count})")
            if original_row_count > 0 :
                 print(f"已筛选出 category 为 {selected_categories} 的行 ({current_row_count}/{original_row_count} 行保留)。")
            else:
                 print(f"已筛选出 category 为 {selected_categories} 的行 (筛选前行数为0)。")
        if min_count_value is not None:
            original_row_count_before_min_count = len(df)
            df = df[df['count'] > min_count_value]
            current_row_count = len(df)
            filters_applied_messages.append(f"'count' > {min_count_value} (筛选前行数: {original_row_count_before_min_count}, 筛选后行数: {current_row_count})")
            if original_row_count_before_min_count > 0:
                print(f"已筛选出 'count' 大于 {min_count_value} 的行 ({current_row_count}/{original_row_count_before_min_count} 行保留)。")
            else:
                print(f"已筛选出 'count' 大于 {min_count_value} 的行 (筛选前行数为0)。")
        if df.empty:
            if filters_applied_messages:
                print(f"在应用以下筛选后，没有数据剩余: {'; '.join(filters_applied_messages)}。")
            else:
                print("原始数据为空或在初始读取后为空。")
            
            try:
                original_columns = pd.read_csv(input_csv_path, nrows=0).columns.tolist()
                empty_df = pd.DataFrame(columns=original_columns)
            except Exception:
                print("警告：无法读取原始文件的列名，将使用默认列['category', 'count']创建空文件")
                empty_df = pd.DataFrame(columns=['category', 'count'])
            
            empty_df.to_csv(output_csv_path, index=False)
            print(f"已生成空的输出文件: {output_csv_path}")
            return
        if n is not None and n > 0:
            print(f"开始按 'category' 分组并选取每组前 {n} 个最大 'count' 的行")
            df_grouped = df.groupby('category', group_keys=False)
            df_filtered = df_grouped.apply(lambda x: x.nlargest(n, 'count'))
        else:
            if n is None:
                print("参数 'n' 未指定，将保留每个 category 中所有符合条件的行，并按 'count' 降序排序")
            else: # n is 0 or negative
                print(f"参数 'n' 为 {n}，将保留每个 category 中所有符合条件的行，并按 'count' 降序排序")
            df_grouped = df.groupby('category', group_keys=False)
            df_filtered = df_grouped.apply(lambda x: x.sort_values('count', ascending=False))
        df_filtered.to_csv(output_csv_path, index=False)
        print(f"处理完成！结果已保存到: {output_csv_path} (共 {len(df_filtered)} 行)")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_csv_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    input_file = 'selected_tags.csv'
    filter_top_n_per_category(input_file, 'cleaned_selected_tags.csv', min_count_value=200)

"""
使用示例
    print("\n--- 示例1: 每个category最多2行，count > 100 ---")
    filter_top_n_per_category(input_file, 'output_n2_min_count100.csv', n=2, min_count_value=100)

    print("\n--- 示例2: 只处理 category 0 和 'A'，每个category最多1行，count > 60 ---")
    filter_top_n_per_category(input_file, 'output_selected_cat_0_A_n1_min60.csv', n=1, min_count_value=60, selected_categories=[0, 'A'])

    print("\n--- 示例3: 处理所有 category，不限制行数 (n=None)，count > 90 ---")
    filter_top_n_per_category(input_file, 'output_all_rows_min_count90.csv', n=None, min_count_value=90)
    # 或者 filter_top_n_per_category(input_file, 'output_all_rows_min_count90.csv', min_count_value=90) # n 默认就是 None

    print("\n--- 示例4: 处理 category 1，不限制行数 (n=0)，无count限制 ---")
    filter_top_n_per_category(input_file, 'output_cat1_all_rows_n0.csv', n=0, selected_categories=[1])

    print("\n--- 示例5: 处理所有 category，每个category最多3行，count > 0 ---")
    filter_top_n_per_category(input_file, 'output_all_cat_n3_min0.csv', n=3, min_count_value=0)
    
    print("\n--- 示例6: 只处理 category 2，但 min_count_value 很高导致无数据剩余 ---")
    filter_top_n_per_category(input_file, 'output_selected_cat_2_no_data.csv', n=2, min_count_value=1000, selected_categories=[2])

    print("\n--- 示例7: 处理不存在的 category，n=None ---")
    filter_top_n_per_category(input_file, 'output_non_existent_cat_all_rows.csv', n=None, selected_categories=[99, 'Z'])
    
    print("\n--- 示例8: 默认行为 (n=None, min_count_value=None, selected_categories=None) - 即所有数据按category分组后按count降序 ---")
    filter_top_n_per_category(input_file, 'output_default_behavior.csv')
"""
