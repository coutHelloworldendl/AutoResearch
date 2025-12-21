from utils import Field_Selector
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conference_name", type=str, required=True, help="Name of the conference")
    parser.add_argument("--year", type=int, required=True, help="Year of the conference")
    args = parser.parse_args()

    field_selector = Field_Selector(conference_name=args.conference_name, year=args.year)
    field_selector.forward()
