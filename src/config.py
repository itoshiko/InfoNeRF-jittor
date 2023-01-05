import toml

def load_config(cfg_path, base_path=None):
    cfg = {}
    if base_path is None:
        return toml.load(cfg_path)
    cfg = toml.load(base_path)
    cfg_ex = toml.load(cfg_path)
    return merge_dict(cfg, cfg_ex)


def merge_dict(dict1, dict2):
    result = dict(dict1)
    for i in dict2:
        if i in result:
            if type(result[i]) is dict and type(dict2[i]) is dict:
                result[i] = merge_dict(result[i], dict2[i])
            else:
                result[i] = dict2[i]
        else:
            result[i] = dict2[i]
    return result


if __name__ == '__main__':
    cfg = load_config('configs/lego.toml', 'configs/base.toml')
    print(cfg)