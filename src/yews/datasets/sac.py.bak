import numpy as np
#import obspy

from . import utils
from .dirs import DatasetFolder

def load_three_component_sac(sample):
    """Read a seismic frame given a sample from datasets.

    """
    path, starttime, endtime = sample
    return utils.read_frame_obspy(path, starttime=starttime, endtime=endtime)


class MarianaFromSource(DatasetFolder):
    """`Mariana <https://arxiv.org/abs/1901.06396>`_ Dataset stored in folder.

        root/.../event_p/sta-year_julday_hr_min_sec_msec-mag.BHX
        root/.../event_p/sta-year_julday_hr_min_sec_msec-mag.BHY
        root/.../event_p/sta-year_julday_hr_min_sec_msec-mag.BHZ

    Args:
        path (str): Root directory where ``wenchuan/processed/data.npy`
            and ``wenchuan/processed/labels.npy`` exists.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    """

    def __init__(self, **kwargs):
        self.raw_samples = None
        super().__init__(loader=load_three_component_sac, **kwargs)

    def build_dataset(self):
        """Construct samples list and targets list.

        """

        # get file patterns and arrival times
        files = self.get_files_under_dir(self.root, '**/*.BH*')
        patterns = [p.parent / (p.name.split('.BH')[0] + '.BH*') for p in files]
        patterns = list(set([p for p in patterns if patterns.count(p) == 3]))
        arrivals = [p.name.split('-')[2].split('_') for p in patterns]
        arrivals = [f"{c[0]}-{c[1]}T{c[2]}:{c[3]}:{c[4]}" for c in arrivals]

        # construct samples and labels
        samples = []
        targets = []
        for i in range(len(patterns)):
            starttime = utils.UTCDateTime(arrivals[i]) - 5
            endtime = utils.UTCDateTime(arrivals[i]) + 15
            filename = str(patterns[i])

            # phase
            target = {
                'event_p': 1,
                'event_s': 2,
            }.get(patterns[i].parent.name)

            samples.append((filename, starttime, endtime))
            targets.append(target)

            # noise
            offset = -75 if target==1 else 65
            starttime += offset
            endtime += offset
            target = 0

            samples.append((filename, starttime, endtime))
            targets.append(target)

        self.raw_samples = samples
        samples = self.FilesLoader(samples, self.loader)
        targets = np.array(targets)
        return samples, targets


def load_three_component_sac_resample(sample):
    path, starttime, endtime = sample
    st = obspy.read(path, starttime=starttime, endtime=endtime)
    fs = 100
    st.resample(fs)
    return utils.stream2array(st)[:, :2000]


class OKFromSource(DatasetFolder):
    """`OK <https://arxiv.org/abs/1901.06396>`_ Dataset stored in folder.

        root/event_id/phase-HHE-arrival_time-mag.SAC
        root/event_id/phase-HHN-arrival_time-mag.SAC
        root/event_id/phase-HHZ-arrival_time-mag.SAC

    Args:
        path (str): Root directory where ``wenchuan/processed/data.npy`
            and ``wenchuan/processed/labels.npy`` exists.
        sample_transform (callable, optional): A function/transform that takes
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            a target and transform it.

    """

    def __init__(self, pattern='**/*.SAC', **kwargs):
        self.raw_samples = None
        self.pattern = pattern
        super().__init__(loader=load_three_component_sac_resample, **kwargs)

    def build_dataset(self):
        files = get_files_under_dir(self.root , self.pattern)
        patterns = []
        for f in files:
            parent = f.parent
            name = f.name.split('.', 3)
            cha = name[2][:2] + '*'
            name[2] = cha
            name = '.'.join(name)
            patterns.append(parent / name)
        patterns = list(set([p for p in patterns if patterns.count(p) == 3]))

        # construct samples and labels
        samples = []
        targets = []
        for p in patterns:
            filename = str(p)
            header = obspy.read(filename.replace('*', 'Z'), headonly=True)[0].stats
            starttime = header['starttime']
            endtime = header['endtime']
            dur =  endtime - starttime
            phase_starttime = starttime + 115 * (dur / 240)
            phase_endtime = starttime + 135 * (dur / 240)

            # phase
            target = {
                'P': 1,
                'S': 2,
            }.get(p.name.split('.')[0])

            samples.append((filename, phase_starttime, phase_endtime))
            targets.append(target)

            # noise
            offset = -75 * (dur / 240) if target==1 else 65 * (dur / 240)
            noise_starttime = phase_starttime + offset
            noise_endtime = phase_endtime + offset
            target = 0

            samples.append((filename, noise_starttime, noise_endtime))
            targets.append(target)

        self.raw_samples = samples
        samples = self.FilesLoader(samples, self.loader)
        targets = np.array(targets)
        return samples, targets
