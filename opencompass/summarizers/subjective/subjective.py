# flake8: noqa: E501
import csv
import os
import os.path as osp
import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from mmengine import ConfigDict

from .alignmentbench import AlignmentBenchSummarizer
from .alpacaeval import AlpacaSummarizer
from .arenahard import ArenaHardSummarizer
from .compass_arena import CompassArenaSummarizer

class SubjectiveSummarizer:
    """Do the subjectivity analyze based on evaluation results.

    Args:
        config (ConfigDict): The configuration object of the evaluation task.
            It's expected to be filled out at runtime.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.cfg = config

    def summarize(self,
                  time_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')):
        """Summarize the subjectivity analysis based on evaluation results.

        Args:
            time_str (str): Timestamp for file naming.

        Returns:
            pd.DataFrame: The summary results.
        """
        dataset_cfgs = self.cfg['datasets']
        for judge_model in self.judge_models:
            judge_abbr = model_abbr_from_cfg(judge_model)
            dataset_cfgs = self.cfg['datasets']
            output_dir, results_folder = get_outdir(self.cfg, time_str)
            fout_flag, fout_flag2 = 0, 0
            for eval_model_abbr in self.eval_model_abbrs:
                subdir = eval_model_abbr + '_judged-by--' + judge_abbr
                subdir_path = os.path.join(results_folder, subdir)
                if os.path.isdir(subdir_path):
                    model = eval_model_abbr
                    if self.judge_type == 'general':
                        fout = osp.join(
                            output_dir,
                            'judged-by--' + judge_abbr + '-dimension.csv')
                    fout2 = osp.join(
                        output_dir,
                        'judged-by--' + judge_abbr + '-capability.csv')
                    for dataset in dataset_cfgs:
                        judged_answers, references = get_judgeanswer_and_reference(
                            dataset, subdir_path, self.judge_function)
                        if self.judge_type == 'general':
                            get_dimension_results(judged_answers, references,
                                                  fout, fout_flag, model)
                            fout_flag += 1
                        get_capability_results(judged_answers, references,
                                               fout2, fout_flag2, model,
                                               self.category)
                        fout_flag2 += 1
                else:
                    print(subdir_path + ' is not exist! please check!')
        if self.judge_type == 'general':
            with open(fout, 'r') as f:
                x = from_csv(f, delimiter=',')
            print(x)
            print(fout)
        with open(fout2, 'r') as f:
            x = from_csv(f, delimiter=',')
        print(x)
        print(fout2)


