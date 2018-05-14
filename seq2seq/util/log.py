from __future__ import print_function

import torch

from collections import defaultdict

class Log(object):
    """
    The Log can be used to store logs during training, write the to a file
    and read them again later.
    """

    def __init__(self, path=None):
        self.steps = []
        self.log = defaultdict(lambda: defaultdict(list))

        if path is not None:
            self.read_from_file(path)

    def write_to_log(self, dataname, losses, metrics, step):
        """
        Add new losses to Log object.
        """
        for metric in metrics:
            val = metric.get_val()
            self.log[dataname][metric.log_name].append(val)

        for loss in losses:
            val = loss.get_loss()
            self.log[dataname][loss.log_name].append(val)

    def update_step(self, step):
        self.steps.append(step)

    def write_to_file(self, path):
        """
        Write the contents of the log object to a file. Format:

        steps step1 step2 step3 ...
        name_of_dataset1
            metric_name val1 val2 val3 val4 ...
            loss_name val1 val2 val3 val4 ...
            ...
        name_of_dataset2
            ...
        """

        f = open(path, 'wb')

        # write steps
        steps = "steps %s\n" % ' '.join(['%i' % step for step in self.steps])
        f.write(steps.encode())

        # write logs
        for dataset in self.log.keys():
            f.write(dataset.encode()+b'\n')
            for metric in self.log[dataset]:
                log = "\t%s %s\n" % (metric, ' '.join([str(v) for v in self.log[dataset][metric]]))
                f.write(log.encode())

        f.close()

    def read_from_file(self, path):
        """
        Fill the contents of a log object reading information
        from a file that was also generated by a log object.
        The format of this file should be:

        steps step1 step2 step3 ...
        name_of_dataset1
            metric_name val1 val2 val3 val4 ...
            loss_name val1 val2 val3 val4 ...
            ...
        name_of_dataset2
            ...
        """
        f = open(path, 'rb')

        lines = f.readlines()
        self.steps = [int(i) for i in lines[0].split()[1:]]

        for line in lines[1:]:
            l_list = line.split()
            if len(l_list) == 1:
                cur_set = l_list[0].decode()
            else:
                data = [float(i) for i in l_list[1:]]
                self.log[cur_set][l_list[0].decode()] = data

    def get_logs(self):
        return self.log

    def get_steps(self):
        return self.steps