import yaml
from pathlib import Path
from config.schema import Config, TrainingConfig, TestConfig, PathConfig, SplitConfig

def load_train_config(config_path: Path) -> Config:
    """
    モデルのトレーニング用の設定をYAMLファイルから読み込みます。
    
    引数:
    - config_path (Path): 設定を含むYAMLファイルへのパス。
    
    戻り値:
    - config (Config): 読み込まれた設定。
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        
    training_config = TrainingConfig(**config_dict['training'])
    path_config = PathConfig(**config_dict['paths'])
    split_config = SplitConfig(**config_dict['splits'])
    
    return Config(training=training_config, paths=path_config, splits=split_config)

def load_test_config(config_path: Path) -> Config:
    """
    モデルのテスト用の設定をYAMLファイルから読み込みます。
    
    引数:
    - config_path (Path): 設定を含むYAMLファイルへのパス。
    
    戻り値:
    - config (Config): 読み込まれた設定。
    """
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)
        
    test_config = TestConfig(**config_data['test'])
    path_config = PathConfig(**config_data['paths'])
    split_config = SplitConfig(**config_data['splits'])
    
    return Config(test=test_config, paths=path_config, splits=split_config)


def main():
    # debug
    train_config = load_train_config(Path('/home/tanaka/0207/config/anomaly_train_config.yaml'))
    
    # configのコンソール出力
    # print(train_config.training)
    # print(train_config.paths)
    # print(train_config.splits.root)
    
    test_config = load_test_config(Path('/home/tanaka/0218/Treatment_CNN/config/anomaly_test_config.yaml'))
    print(test_config.test)
    print(test_config.paths)
    print(test_config.splits.root)
    
if __name__ == '__main__':
    main()