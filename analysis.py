import data_preparation
import models

def main():
    dataset = data_preparation.get_dataset()
    models.run(dataset)


if __name__ == "__main__":
    main()