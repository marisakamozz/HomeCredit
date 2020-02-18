from util import read_all, get_dims, dump

def main():
    all_data = read_all(directory='../data/03_powertransform')
    dims = get_dims(all_data)
    dump(dims, '../data/07_dims/dims03.joblib')
    all_data = read_all(directory='../data/05_onehot')
    dims = get_dims(all_data)
    dump(dims, '../data/07_dims/dims05.joblib')

if __name__ == "__main__":
    main()
