"""Process dataset of simulation results."""

import pathlib
import datetime
from tqdm import tqdm
import pandas as pd
import math
from file_IO_handler import load_json, save_json


def consolidate_jsons_to_mega_json_by_engine_prompt(
    open_folder: pathlib.Path, 
    save_folder: pathlib.Path,
    experiment_descriptor: str
) -> dict:
    """Consolidate .json files by engine and prompt descriptor as properly compressed .json.gz files.

    Args:
        open_folder: folder containing .json files.
        save_folder: folder to save consolidated results as .json.gz files.
        experiment_descriptor: experiment descriptor for output filenames.

    Returns:
        Dictionary with count of files processed for each engine-prompt combination.
    """
    import gzip
    import json
    
    # 确保保存文件夹存在
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    list_of_data_files = list(open_folder.glob("**/*.json"))
    print(f"Found {len(list_of_data_files)} total JSON files in {open_folder}")
    
    # 按引擎和prompt描述符分组文件
    grouped_files = {}
    
    for data_file in list_of_data_files:
        filename = data_file.name
        
        try:
            # 分割文件名
            parts = filename.split('_')
            
            # 找到各个组件的位置
            engine_index = parts.index('engine') + 1
            prompt_index = parts.index('prompt') + 1
            
            engine = parts[engine_index]
            prompt_descriptor = parts[prompt_index]
            
            # 创建分组键
            key = f"{engine}_{prompt_descriptor}"
            
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(data_file)
            
            print(f"File {filename} -> Engine: {engine}, Prompt: {prompt_descriptor}")
            
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse filename {filename}: {e}")
            continue
    
    print(f"\nFound {len(grouped_files)} unique engine-prompt combinations:")
    for key, files in grouped_files.items():
        print(f"  {key}: {len(files)} files")
    
    # 处理每个组合
    consolidation_counts = {}
    
    for key, data_files in grouped_files.items():
        engine, prompt_descriptor = key.split('_', 1)
        
        print(f"\nProcessing {engine} - {prompt_descriptor} ({len(data_files)} files)")
        
        # 读取并合并文件
        mega = []
        for data_file in tqdm(data_files, desc=f"Loading {key}"):
            try:
                file_contents = load_json(data_file)
                mega.append(file_contents)
            except Exception as e:
                print(f"Error loading {data_file}: {e}")
                continue
        
        if mega:
            # 保存为正确压缩的 .json.gz 文件
            save_filename = f"{experiment_descriptor}_{engine}_{prompt_descriptor}.json.gz"
            save_path = save_folder / save_filename
            
            print(f"Saving {len(mega)} records to {save_filename}")
            print(f"Started saving at: {datetime.datetime.now()}")
            
            try:
                # 手动使用 gzip 压缩保存
                with gzip.open(save_path, 'wt', encoding='utf-8') as f:
                    json.dump(mega, f, indent=None, separators=(',', ':'))
                
                print(f"Finished saving at: {datetime.datetime.now()}")
                consolidation_counts[key] = len(data_files)
                print(f"✓ Successfully saved {save_filename}")
                
                # 验证文件是否正确保存和压缩
                try:
                    with gzip.open(save_path, 'rt', encoding='utf-8') as f:
                        test_load = json.load(f)
                    print(f"✓ File verification passed - properly compressed with {len(test_load)} records")
                except Exception as verify_error:
                    print(f"✗ File verification failed: {verify_error}")
                    
            except Exception as e:
                print(f"✗ Error saving {save_filename}: {e}")
                consolidation_counts[key] = 0
        else:
            print(f"✗ No valid data found for {key}")
            consolidation_counts[key] = 0
    
    return consolidation_counts


def consolidate_jsons_to_mega_json(
    open_folder: pathlib.Path, save_file_path: pathlib.Path
) -> int:
    """原始函数保持不变，以便向后兼容"""
    mega = []
    # Convert generator to list to enable len() function
    list_of_data_files = list(open_folder.glob("**/*.json"))  # list .json files in folder

    print(f"Got {len(list_of_data_files)} files in folder {open_folder}")

    for data_file in list_of_data_files:
        print(f"Processing: {data_file}")  # 添加进度信息
        file_contents = load_json(data_file)
        mega.append(file_contents)

    print("Started saving at: ", str(datetime.datetime.now()))
    save_json(mega, save_file_path)
    print("Finished saving at: ", str(datetime.datetime.now()))  # 修正了这里的文字
    return len(list_of_data_files)


def process_mega_jsons(engines: list[str]):
    """Processes mega jsons with improved file path handling.
    Args:
        engines: list of engines
    
    Returns: 
        dict whose keys are engines.
    """
    dict_df_results = dict()
    for engine in engines:
        temp_dfs = {
            prompt_descriptor_accept: None,
            prompt_descriptor_reject: None
        }
        for prompt_descriptor in [prompt_descriptor_accept, prompt_descriptor_reject]:
            
            # 构建文件路径 - 先尝试 .json，然后尝试 .json.gz
            json_path = path_to_simulation_results_consolidated_folder / f"{experiment_descriptor}_{engine}_{prompt_descriptor}.json"
            gz_path = path_to_simulation_results_consolidated_folder / f"{experiment_descriptor}_{engine}_{prompt_descriptor}.json.gz"
            
            # 选择存在的文件
            if json_path.exists():
                file_path = json_path
                print(f"Using JSON file: {json_path.name}")
            elif gz_path.exists():
                file_path = gz_path
                print(f"Using compressed file: {gz_path.name}")
            else:
                print(f"Available files in {path_to_simulation_results_consolidated_folder}:")
                for f in path_to_simulation_results_consolidated_folder.glob("*"):
                    print(f"  {f.name}")
                raise FileNotFoundError(f"Neither {json_path.name} nor {gz_path.name} exists in {path_to_simulation_results_consolidated_folder}")
            
            df = process_mega_json_for_no_complete_prompt(
                path_to_megajson=file_path, 
                completion_is_last_n_tokens_of_echoed_prompt=1,
            )
            print(engine, prompt_descriptor, df["tokens"].value_counts())
            
            # Rename probability and token columns, then drop weird column if it exists.
            rename_dict = {
                "probability": "original p(accept)" if prompt_descriptor == prompt_descriptor_accept else "original p(reject)", 
                "tokens": "token accept" if prompt_descriptor == prompt_descriptor_accept else "token reject"
            }
            df = df.rename(columns=rename_dict)
            
            # 只有在 "Unnamed: 0" 列存在时才删除它
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            
            temp_dfs[prompt_descriptor] = df
            
        # Merge into single dataframe.
        df_result = temp_dfs[prompt_descriptor_accept].merge(
            temp_dfs[prompt_descriptor_reject],
            on=["index", 
                "engine", 
                "money", 
                "keep", 
                "offer", 
                "player1", 
                "player1_surname", 
                "player1_gender", 
                "player1_race",
                "player1_self",
                "player1_poss",
                "player2", 
                "player2_surname", 
                "player2_gender", 
                "player2_race",
                "player2_self",
                "player2_poss"
            ]
        )
        # Get total probability of valid completions.
        df_result["p(valid)"] = df_result["original p(accept)"] + df_result["original p(reject)"]
        df_result["p(invalid)"] = 1 - df_result["p(valid)"]
        # Get normalized probability of accept and reject.
        df_result["p(accept)"] = df_result["original p(accept)"].div(df_result["p(valid)"])
        df_result["p(reject)"] = df_result["original p(reject)"].div(df_result["p(valid)"])
        # derivative columns
        df_result["gender_pair"] = df_result["player1_gender"] + "-" + df_result["player2_gender"]
        df_result["race_pair"] = df_result["player1_race"] + "-" + df_result["player2_race"]
        df_result["name_pair"] = df_result["player1"] + "-" + df_result["player2"]
    
        dict_df_results[engine] = df_result
    return dict_df_results


def process_mega_json_for_no_complete_prompt(
    path_to_megajson: pathlib.Path,
    completion_is_last_n_tokens_of_echoed_prompt: int = 1,
    filter_by_prompt_descriptor: None | str = None,
):
    """Process mega .json.gz file from experiment using a no-complete prompt.

    The `consolidate_jsons_to_mega_json` saves the contents of several .json files to a mega .json.gz file.

    The Ultimatum Game simulation and the Garden Path simulation both used no-complete prompts,
    i.e., no completions were generated by the language model.
    Each experiment was run twice to get the probabilities of the two allowed completions:
        "accept" and "reject" for the Ultimatum Game,
        "grammatical" and "ungrammatical" for the Garden Path.

    Given a mega.json.gz, process its contents from a hierarchial structure to a flat dataframe structure.

    Args:
        path_to_megajson: path to the mega .json.gz file with all the results and information for the experiment.
        completion_is_last_n_tokens_of_echoed_prompt: how many tokens from the end of the prompt to get log probs for
            (completions supplied in prompt).
        filter_by_prompt_descriptor: `filter_by_prompt_descriptor` because different prompts (different completions)
            will have different `log_prob_of_last_n_tokens_in_echoed_prompt`

    Returns:
        pandas dataFrame.

    Raises:
        Exception: When using 'no-complete' type prompt, model settings should include echo.
    """
    # Dict to turn into pandas dataframe.
    results = {"index": [], "engine": [], "tokens": [], "probability": []}
    added_prompt_fill_fields_to_results_dict = False

    mega = load_json(filename=path_to_megajson)
    print(f"Found {len(mega)} items in mega json")

    for res in tqdm(mega):
        # Filter by prompt descriptor and experiment descriptor.
        # Only process results with matching descriptors.
        if (filter_by_prompt_descriptor is not None) and (
            res["input"]["prompt_descriptor"] != filter_by_prompt_descriptor
        ):
            continue

        # Populate field values in results.
        results["index"].append(res["input"]["prompt"]["index"])
        results["engine"].append(res["model"]["engine"])

        # Add prompt fill fields as additional fields in results dict.
        if not added_prompt_fill_fields_to_results_dict:
            for k, v in res["input"]["prompt"]["values"].items():
                results[k] = []
            # Set flag.
            added_prompt_fill_fields_to_results_dict = True

        # Populate prompt fill values in results.
        for k, v in list(res["input"]["prompt"]["values"].items()):
            results[k].append(v)

        # Get log probs for completion supplied in the prompt.
        if not res["model"]["echo"]:
            raise Exception(
                "When using 'no-complete' type prompt, completions should be supplied in prompt so getting probabilites requires echo-ing prompt probabilities."
            )
        # Assume that n = 1 (number of language model responses = 1).
        choice = res["output"]["choices"][0]
        if res["model"]["max_tokens"] == 0:
            # There is no generated text in the output.
            # Prefered way to run 2-choice simulation.
            res["output"]["echo_logprobs"] = choice["logprobs"]
        else:
            # There is generated text in the output (probably by mistake?).
            # This is a problem because completion is counted as tokens from the end of the output.
            # Salvage the run by isolating echo of the prompt from the mistake generation.
            len_input = len(res["input"]["full_input"])
            # Define index to slice with.
            slicer = choice["logprobs"]["text_offset"].index(len_input)
            res["output"]["echo_logprobs"] = {
                "tokens": choice["logprobs"]["tokens"][:slicer],
                "token_logprobs": choice["logprobs"]["token_logprobs"][:slicer],
            }

        # Gather tokens and calculate overall probability for completion.
        tokens_list = []
        logprob_sum = 0
        tokens = res["output"]["echo_logprobs"]["tokens"]
        token_logprobs = res["output"]["echo_logprobs"]["token_logprobs"]
        for i in range(1, completion_is_last_n_tokens_of_echoed_prompt + 1):
            tokens_list.append(tokens[-i])
            logprob_sum += token_logprobs[-i]
        tokens_list.reverse()
        results["tokens"].append("-".join(tokens_list))
        results["probability"].append(math.exp(logprob_sum))

    df_results = pd.DataFrame(results)

    return df_results