class Config:
    use_cache = False
    save_cache = False
    cache_dir = "cache"
    perfect_cluster = False
    perfect_split = False
    perfect_operator = False
    perfect_directory = "data_perfect"
    verbose = False
    money_saver = False
    num_retry = 1

    @staticmethod
    def set_use_cache(value: bool):
        Config.use_cache = value

    @staticmethod
    def set_save_cache(value: bool):
        Config.save_cache = value

    @staticmethod
    def set_cache_dir(value: str):
        Config.cache_dir = value

    @staticmethod
    def set_perfect_cluster(value: bool):
        Config.perfect_cluster = value

    @staticmethod
    def set_perfect_split(value: bool):
        Config.perfect_split = value
    
    @staticmethod
    def set_perfect_operator(value: bool):
        Config.perfect_operator = value

    @staticmethod
    def set_perfect_directory(value: str):
        Config.perfect_directory = value

    @staticmethod
    def set_verbose(value: bool):
        Config.verbose = value

    @staticmethod
    def set_money_saver(value: bool):
        Config.money_saver = value