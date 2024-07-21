import os
import sys
from utils import DataWrapper, write_config

TRAIN_SIZE = 0.7
VALID_SIZE = 0.1

TARGET_COLUMN = "measured_t"

def prepare_data(data_path: str, output_dirpath: str):
    configs = {}
    df = DataWrapper(data_path, output_dirpath)

    configs['cat_feats'] = []
    cat_feats = df.get_categorical_columns()
    cat_feats = list(set(cat_feats) - set(["date", "time", "tmstamp"]))
    for cat_feat in cat_feats:
        df.label_encoding(cat_feat)
        #print(f"Column: {cat_feat}, Uniques: {df.df[cat_feat].nunique()}")
        configs['cat_feats'].append(
            {
                "column_name" : cat_feat,
                "num_uniques": int(df.df[cat_feat].nunique())
            }
        )

    # garbage column
    if 'tracker2wm_avg' in df.df.columns:
        df.df.drop(columns=['tracker2wm_avg'], inplace=True)

    cont_feats = df.get_continuous_numeric_columns()
    cont_feats = list(set(cont_feats) - set(['device_code', TARGET_COLUMN]) - set(cat_feats))
    
    # no normalizing of preprocessing of the continious
    configs["num_feats"] = cont_feats

    for col in df.get_nan_containing_columns():
        method = None
        if col in cont_feats:
            method = 'mean'
        elif col in cat_feats:
            method = "most_freq"
        else:
            print("Unknown column for the NaN fill: ", col)
            continue
        df.fillna(col, method=method)

    df.slice_sequential(train_prop=TRAIN_SIZE, valid_prop=VALID_SIZE)

    df.save_train_df()
    df.save_valid_df()
    df.save_test_df()

    configs["target_feature"] = TARGET_COLUMN
    write_config(os.path.join(output_dirpath, "configs.yaml"), configs)

if __name__ == "__main__":
    i_path = sys.argv[1]
    o_path = sys.argv[2]
    prepare_data(i_path, o_path)