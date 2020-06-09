from model import build_model
from utils import parse_args
from benchmark import benchmark_run


def main():
    args = parse_args()
    model, is_3d = build_model(args.model, args.opts)
    benchmark_run(model, is_3d)


if __name__ == '__main__':
    main()
