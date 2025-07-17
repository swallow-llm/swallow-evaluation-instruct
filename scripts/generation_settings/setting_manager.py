from argparse import ArgumentParser
from pathlib import Path
import csv
import yaml

SUPPORTED_PARAMS = [
    "system_message",
    "max_model_length",
    "max_new_tokens",
    "temperature",
    "top_p",
    "max_n",
    "reasoning_parser",
    "vllm_use_v1",
]

CONFIG_YAML_PARAMETERS = [
    # These parameters will be printed in the 'generation' section of the config (.yaml) file for litellm.
    # If not specified, the parameter will be skipped (and use the default value specified by litellm).
    "max_new_tokens",
    "temperature",
    "top_p",
    "max_n",
]

SHELL_OUTPUT_PARAMETERS = {
    # These parameters will be output in the standard output in the order of the list.
    # The order must match the order of the parameters in get_generation_params().
    "max_model_length": -1, # used to serve a LLM with vllm
    "reasoning_parser": "", # used to serve a LLM with vllm
    "vllm_use_v1": 1,       # used to serve a LLM with vllm
    "system_message": "",   # used to launch evaluation with lighteval
}

assert set(SUPPORTED_PARAMS) == set(CONFIG_YAML_PARAMETERS+list(SHELL_OUTPUT_PARAMETERS.keys())), "💀 CONFIG_YAML_PARAMETERS and SHELL_OUTPUT_PARAMETERS must be a complete subset of SUPPORTED_PARAMS."


class SettingManager:
    def __init__(self, custom_model_settings_dir: str, task_config_path: str, verbose: bool = False):
        # load model configs (yaml files) in the dir
        self.custom_model_settings_dir = custom_model_settings_dir
        self.custom_model_settings = {}
        self.custom_model_settings_paths = {}
        for publisher_dir in Path(custom_model_settings_dir).iterdir():
            if not publisher_dir.is_dir():
                continue
            publisher_name = publisher_dir.name
            for yaml_file in publisher_dir.glob("*.yaml"):
                model_name = yaml_file.stem
                with open(yaml_file, "r") as f:
                    settings = yaml.load(f, Loader=yaml.FullLoader)
                self.custom_model_settings[f"{publisher_name}_{model_name}"] = settings
                self.custom_model_settings_paths[f"{publisher_name}_{model_name}"] = yaml_file

        # load task config (csv file)
        self.task_config_path = task_config_path
        self.task_config = {}
        with open(self.task_config_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == "key": continue
                self.task_config[row[0]] = row[1]
        self.task_config_keys = list(self.task_config.keys())

        # verbose
        self.verbose = verbose


    def search_model_settings(self, model_id: str, custom_settings: str) -> dict:
        # If model found and settings found -> return the settings. Otherwise, raise an error.
        # So, if the function terminates successfully, the desired settings should be found.

        assert model_id != "", "💀 Model ID is not specified."
        assert custom_settings != "", "💀 Custom settings is not specified."

        model_id_key = model_id.replace("/", "_")
        model_setting_dict = self.custom_model_settings.get(model_id_key, {})

        if isinstance(model_setting_dict, dict) and len(model_setting_dict.keys())>0:
            # Found candidate settings
            model_settings = model_setting_dict.get(custom_settings, {})
        elif model_setting_dict == {}:
            # No settings defined for the specified model
            assert False, f"💀 Custom settings for \'{model_id}\' not found in \'{self.custom_model_settings_dir}\'."
        else:
            # Nothing defined in the yaml file or defined in a wrong format
            assert False, f"💀 Custom settings for \'{model_id}\' might be defined without any settings or in a wrong format."

        if model_settings == {}:
            # Custom settings not found in the defined settings
            assert False, f"💀 Custom settings \'{custom_settings}\' not found in defined settings, {list(model_setting_dict.keys())}."
        else:
            # Custom settings found in the defined settings
            if self.verbose: print(f"✅ Custom settings, \'{custom_settings}\' for \'{model_id}\' was found.")
            return model_settings


    def search_task_settings(self, task_key: str) -> dict:
        task_settings_oneline = self.task_config.get(task_key, "notfound")
        assert task_settings_oneline != "notfound", f"💀 Task settings for \'{task_key}\' not found in \'{self.task_config_path}\'."
        
        if self.verbose: print(f"✅ Task settings for \'{task_key}\' was found.")
        task_settings = {}
        for setting in task_settings_oneline.split(","):
            if "=" in setting:
                param, value = setting.split("=")
                task_settings[param] = value
        return task_settings


    def merge_settings(self, model_settings: dict, task_settings: dict, merge_strategy: str) -> dict:
        merged_settings = {}
        if self.verbose: print(f"🔍 Merging settings with merge strategy: {merge_strategy}")
        
        strategies = {
            "model-first": (model_settings, task_settings),
            "task-first": (task_settings, model_settings),
        }
        if merge_strategy not in strategies:
            raise ValueError(f"Invalid merge strategy: {merge_strategy}")
        primary, secondary = strategies[merge_strategy]

        for param in SUPPORTED_PARAMS:
            for src in (primary, secondary):
                if param in src:
                    if src[param] is None: continue # No specification of the parameter
                    merged_settings[param] = src[param]
                    break
        return merged_settings
    

    def extract_generation_params(self, settings: dict) -> str:
        generation_params = ""
        for param, value in settings.items():
            if param in CONFIG_YAML_PARAMETERS:
                generation_params += f"""        {param}: {value}
"""
        if len(generation_params) > 0:
            generation_params = f"""    generation:
{generation_params}"""
        return generation_params

    
    def serch_settings_shell(self, model_id: str, task_id: str, custom_settings: str, merge_strategy: str):
        # This function is supposed to be used in shell script. 
        # So the output must be printed as standard output in the following order:
        # - CUSTOM_SETTINGS_PATH: path to the custom settings
        # - CUSTOM_SETTINGS_VERSION: version of the custom settings
        # - SYSTEM_MESSAGE: system message
        # - MAX_MODEL_LENGTH: max model length
        # - GENERATION_PARAMS: generation parameters


        # try to find the custom model settings
        if custom_settings == "":
            model_settings = {}
            custom_settings_path = ""
            custom_settings_version = ""
        else:
            model_settings = self.search_model_settings(model_id, custom_settings)
            custom_settings_path = self.custom_model_settings_paths.get(model_id.replace("/", "_"), "")
            custom_settings_version = model_settings.get("version", "")
        
        # try to find the task settings
        task_settings = self.search_task_settings(task_id)

        # merge the settings
        merged_settings = self.merge_settings(model_settings, task_settings, merge_strategy)

        # extract the generation parameters
        generation_params = self.extract_generation_params(merged_settings)

        # print the results (output for shell script)
        print(custom_settings_path)
        print(custom_settings_version)
        for key in SHELL_OUTPUT_PARAMETERS:  # print the settings each by one line
            if key in merged_settings:
                print(merged_settings[key])
            else:
                print(SHELL_OUTPUT_PARAMETERS[key])
        print(generation_params)    # print the generation parameters at the end


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--custom_settings", type=str, required=True)
    parser.add_argument("--task_settings_path", type=str, default="scripts/generation_settngs/task_settings.csv")
    parser.add_argument("--custom_model_settings_dir", type=str, default="scripts/generation_settngs/custom_model_settings")
    parser.add_argument("--merge_strategy", choices=["task-first", "model-first"], default="model-first")
    parser.add_argument("--verbose", action="store_true")   # Use only for debugging
    args = parser.parse_args()
    
    setting_manager = SettingManager(args.custom_model_settings_dir, args.task_settings_path, args.verbose)
    setting_manager.serch_settings_shell(args.model_id, args.task_id, args.custom_settings, args.merge_strategy)
