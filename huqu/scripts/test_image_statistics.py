from datasets import load_dataset, get_dataset_config_names
from huqu.image_statistics import DatasetAnalyzer

def test_dataset_analyzer():
    # Dataset paths
    HARD_HAT_DETECTION = "keremberke/hard-hat-detection"
    CPPE_5 = "rishitdagli/cppe-5"
    GERMAN_TRAFFIC_SIGN = "keremberke/german-traffic-sign-detection"
    TABLE_EXTRACTION = "keremberke/table-extraction"
    DOCLAYNET_SMALL = "pierreguillou/DocLayNet-small"
    ASBESTOS = "Francesco/asbestos"

    # Hard Hat Detection dataset
    print(f"\nLoading {HARD_HAT_DETECTION} dataset...")
    configs = get_dataset_config_names(HARD_HAT_DETECTION)
    config_name = configs[0] if configs else None
    hard_hat_dataset = load_dataset(HARD_HAT_DETECTION, name=config_name)
    hard_hat_analyzer = DatasetAnalyzer(hard_hat_dataset, image_key="image")
    hard_hat_analyzer.analyze()
    hard_hat_analyzer.report()

    # CPPE-5 dataset
    print(f"\nLoading {CPPE_5} dataset...")
    configs = get_dataset_config_names(CPPE_5)
    config_name = configs[0] if configs else None
    cppe_dataset = load_dataset(CPPE_5, name=config_name)
    cppe_analyzer = DatasetAnalyzer(cppe_dataset, image_key="image")
    cppe_analyzer.analyze()
    cppe_analyzer.report()

    # German Traffic Sign dataset
    print(f"\nLoading {GERMAN_TRAFFIC_SIGN} dataset...")
    configs = get_dataset_config_names(GERMAN_TRAFFIC_SIGN)
    config_name = configs[0] if configs else None
    traffic_dataset = load_dataset(GERMAN_TRAFFIC_SIGN, name=config_name)
    traffic_analyzer = DatasetAnalyzer(traffic_dataset, image_key="image")
    traffic_analyzer.analyze()
    traffic_analyzer.report()

    # Table Extraction dataset
    print(f"\nLoading {TABLE_EXTRACTION} dataset...")
    configs = get_dataset_config_names(TABLE_EXTRACTION)
    config_name = configs[0] if configs else None
    table_dataset = load_dataset(TABLE_EXTRACTION, name=config_name)
    table_analyzer = DatasetAnalyzer(table_dataset, image_key="image")
    table_analyzer.analyze()
    table_analyzer.report()

    # Asbestos dataset
    print(f"\nLoading {ASBESTOS} dataset...")
    configs = get_dataset_config_names(ASBESTOS)
    config_name = configs[0] if configs else None
    asbestos_dataset = load_dataset(ASBESTOS, name=config_name)
    asbestos_analyzer = DatasetAnalyzer(asbestos_dataset, image_key="image")
    asbestos_analyzer.analyze()
    asbestos_analyzer.report()

if __name__ == "__main__":
    test_dataset_analyzer() 