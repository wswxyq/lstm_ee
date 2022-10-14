import argparse
import pickle
import os

from lstm_ee.presets import PRESETS_EVAL
from lstm_ee.utils.eval import standard_eval_prologue
from lstm_ee.utils.log import setup_logging
from lstm_ee.utils.parsers import add_basic_eval_args, add_concurrency_parser
from lstm_ee.eval.predict import get_true_energies, get_base_energies, predict_energies

# usage: python scripts/eval/testdata_gen.py --preset dune_numu_5GeV NETWORK_PATH

def parse_cmdargs():
    # pylint: disable=missing-function-docstring
    parser = argparse.ArgumentParser("Make Binstat plots")

    add_basic_eval_args(parser, PRESETS_EVAL)
    add_concurrency_parser(parser)

    return parser.parse_args()


def main():
    # pylint: disable=missing-function-docstring
    setup_logging()
    cmdargs = parse_cmdargs()

    dgen, args, model, _outdir, plotdir, eval_specs = standard_eval_prologue(
        cmdargs, PRESETS_EVAL
    )

    pred_model_dict = predict_energies(args, dgen, model)
    true_dict = get_true_energies(dgen)

    print('Saving the objects...')
    with open(os.path.join(_outdir, "pred_true_dic.pkl"), "wb") as f:
        pickle.dump([pred_model_dict, true_dict], f)
    print('Saved pred_true_dic.pkl to {}'.format(_outdir))


if __name__ == "__main__":
    main()
