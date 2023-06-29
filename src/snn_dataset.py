import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy


def construct_column_names(config):
    columns = []
    x_col = []
    y_col = []
    for i in range(config['x_size'][0]):
        x_col.append("x" + str(i))
        columns.append("x" + str(i))
    for i in range(config['y_size'][0]):
        y_col.append("y" + str(i))
        columns.append("y" + str(i))
    return columns, x_col, y_col


def get_columns():
    return ['Environment', 'x_size', 'y_size', 'number_episodes', 'episode_length', 'train_size', 'validation_size', 'test_size']


def get_config(directory, config_columns):
    config = pd.read_csv(directory+'config.csv', names=config_columns, na_values="?", comment='\t', sep=",", skipinitialspace=True)
    columns, x_col, y_col = construct_column_names(config=config)
    return config, columns, x_col, y_col


def read_data(path, column_names):
    raw_data = pd.read_csv(path, names=column_names, na_values="?", comment='\t', sep=",", skipinitialspace=False)
    return raw_data.copy()


def load_data(directory, data, data_name):
    config_columns = get_columns()
    config, columns, x_col, y_col = get_config(directory, config_columns)
    eps_length = config.get("episode_length")[0]
    data = data.append(read_data(path=directory + data_name + ".csv", column_names=columns))
    data_x, data_y = split_variables(dataset=data, x_labels=x_col, y_labels=y_col)

    return data, data_x, data_y, columns, x_col, y_col, eps_length


def split_variables(dataset, x_labels, y_labels):
    '''
    Sequential Splitting of variables
    :param dataset:
    :param x_labels:
    :param y_labels:
    :return:
    '''
    #print("Dataset type = ", type(dataset))
    x = dataset.loc[:, x_labels]
    #print("Dataset x type = ", type(x))
    #print(x)
    y = dataset.loc[:, y_labels]
    #print("Dataset y type = ", type(y))
    #print(y)
    return x, y


def calc_stats(data_x, data_y):
    '''
    Normalization may result in a 0 denominator when dividing, ensure denominators are all non-zero
    :param data_x:
    :param data_y:
    :return:
    '''
    mean_x = numpy.mean(data_x).values
    mean_y = numpy.mean(data_y).values
    std_x = numpy.std(data_x).values
    std_x = numpy.where(std_x == 0.0, 0.000000001, std_x)
    std_y = numpy.std(data_y).values
    std_y = numpy.where(std_y == 0.0, 0.000000001, std_y)
    min_x = numpy.min(data_x).values
    max_x = numpy.max(data_x).values
    min_y = numpy.min(data_y).values
    max_y = numpy.max(data_y).values
    #print("X Min = ", min_x)
    #print("X Mean = ", mean_x)
    #print("X Max = ", max_x)
    #print("X Std = ", std_x)
    #print("Y Min = ", min_y)
    #print("Y Mean = ", mean_y)
    #print("Y Max = ", max_y)
    #print("Y Std = ", std_y)
    return mean_x, std_x, mean_y, std_y, min_x, max_x, min_y, max_y


#standardization, min_max
class SNN_Dataset(Dataset):
    # Download/Read Data, etc
    def __init__(self, directory, type_set, normalization_type="normal"):
        self.data_name = type_set
        self.config, columns, x_col, y_col = get_config(directory, get_columns())
        self.data = pd.DataFrame(columns=columns)
        self.data, data_x, data_y, self.columns, self.x_col, self.y_col, self.eps_length = load_data(directory, self.data, self.data_name)
        self.x_size = len(self.x_col)
        self.y_size = len(self.y_col)
        #print("Hello There0 = ", data_x)
        #print("General Kenobi0 = ", type(data_x))
        #print("Hello There = ", data_x.values)
        #print("General Kenobi = ", type(data_x.values))
        self.data_x = torch.tensor(data_x.values)
        self.data_y = torch.tensor(data_y.values)
        self.normalization_type = normalization_type
        self.mean_x, self.std_x, self.mean_y, self.std_y, self.min_x, self.max_x, self.min_y, self.max_y = calc_stats(data_x, data_y)


    # Return the data length
    def __len__(self):
        return len(self.data)

    # Return one Item on the Index, idx
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''x = self.data_x.iloc[idx]
        y = self.data_y.iloc[idx]
        if self.transform:
            x = self.transform(x)
        return x, y'''
        return self.data_x[idx], self.data_y[idx]

    def get_stats(self):
        return self.min_x, self.mean_x, self.max_x, self.std_x, self.min_y, self.mean_y, self.max_y, self.std_y


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


class SNN_Dataset_PyBullet(Dataset):
    # Download/Read Data, etc
    def __init__(self, directory, type, normalization_type="normal"):
        self.data_name = type
        self.config, columns, x_col, y_col = get_config(directory, get_columns())
        self.data = pd.DataFrame(columns=columns)
        self.data, data_x, data_y, self.columns, self.x_col, self.y_col, self.eps_length = load_data(directory, self.data, self.data_name)
        self.x_size = len(self.x_col)
        self.y_size = len(self.y_col)
        self.data_x = torch.tensor(data_x.values)  # torch.from_numpy(data_x.values)
        self.data_y = torch.tensor(data_y.values)#torch.from_numpy(data_y.values)
        self.normalization_type = normalization_type
        self.mean_x, self.std_x, self.mean_y, self.std_y, self.min_x, self.max_x, self.min_y, self.max_y = calc_stats(data_x, data_y)


        # Return the data length
    def __len__(self):
        return len(self.data)

    def get_stats(self):
        return self.min_x, self.mean_x, self.max_x, self.std_x, self.min_y, self.mean_y, self.max_y, self.std_y
        #return self.x_min.values, self.x_mean.values, self.x_max.values, self.x_std.values, self.y_min.values, self.y_mean.values, self.y_max.values, self.y_std.values

    # Return one Item on the Index, idx
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        '''x = self.data_x.iloc[idx]
        y = self.data_y.iloc[idx]
        if self.transform:
            x = self.transform(x)
        return x, y'''
        return self.data_x[idx], self.data_y[idx]


def normalization(x, normalization_type=None, x_min=0, x_mean=0, x_max=0, x_std=0):
    #print("Type ", normalization_type)
    #print("val = ", x)
    #print("min = ", x_min)
    #print("mean = ", x_mean)
    #print("max = ", x_max)
    #print("std = ", x_std)
    #print("Dimensions = ", x.shape)
    x_out = []
    #print("Type x = ", type(x))
    if "standardization" in normalization_type:
        if x.ndim > 1:
            for row in x:
                x_procesessing = standard_score(row, x_mean, x_std)
                x_out.append(x_procesessing)
            x_out = numpy.array(x_out)
            return x_out
        else:
            return standard_score(x, x_mean, x_std)
    elif "min_max" in normalization_type:
        if x.ndim > 1:
            for row in x:
                x_procesessing = min_max(row, x_min, x_max)
                x_out.append(x_procesessing)
            x_out = numpy.array(x_out)
            return x_out
        else:
            return min_max(x, x_min, x_max)
    else:
        return x


def reverse_normalization(x, normalization_type=None, x_min=0, x_mean=0, x_max=0, x_std=0):
    x_out = []

    if (len(x) != len(x_mean)):
        print("Reverse Standardization")
        print("len(x) = ", len(x), " ________ ", "len(x_mean) = ", len(x_mean))
    if "standardization" in normalization_type:
        if x.ndim > 1:
            for row in x:
                x_procesessing = reverse_standard_score(row, x_mean, x_std)
                x_out.append(x_procesessing)
            x_out = numpy.array(x_out)
            return x_out
        else:
            return reverse_standard_score(x, x_mean, x_std)
    elif "min_max" in normalization_type:
        if x.ndim > 1:
            for row in x:
                x_procesessing = reverse_min_max(row, x_min, x_max)
                x_out.append(x_procesessing)
            x_out = numpy.array(x_out)
            return x_out
        else:
            return reverse_min_max(x, x_min, x_max)
    else:
        return x

def standard_score(i, mean_i, std_i):
    return (i - mean_i) / std_i


def reverse_standard_score(i_, mean, std):
    return (i_ * std) + mean


def reverse_min_max(i_, min, max):
    return (i_ * (max - min)) + min


def min_max(i, min_i, max_i):
    return (i - min_i)/(max_i - min_i)
