from copy import copy
from glob import glob
import math
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from ventmap.raw_utils import read_processed_file


class ARDSDataset(Dataset):
    def __init__(self, seq_len, dataset_type, to_pickle=None):
        """
        Dataset to generate sequences of data for ARDS Detection
        """
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        cohort_file = os.path.join(data_path, 'anon-desc.csv')
        self.seq_len = seq_len
        self.n_sub_batches = 20
        self.all_sequences = []

        self.cohort = pd.read_csv(cohort_file)
        self.cohort = self.cohort.rename(columns={'Patient Unique Identifier': 'patient_id'})
        self.cohort['patient_id'] = self.cohort['patient_id'].astype(str)

        raw_dir = os.path.join(data_path, 'experiment1', dataset_type, 'raw')
        if not os.path.exists(raw_dir):
            raise Exception('No directory {} exists!'.format(raw_dir))
        self.raw_files = sorted(glob(os.path.join(raw_dir, '*/*.raw.npy')))
        self.processed_files = sorted(glob(os.path.join(raw_dir, '*/*.processed.npy')))
        self.get_dataset()
        self.derive_scaling_factors()
        if to_pickle:
            pd.to_pickle(self, to_pickle)

    def derive_scaling_factors(self):
        indices = [range(len(self.all_sequences))]
        self.scaling_factors = {
            None: self._get_scaling_factors_for_indices(idxs)
            for i, idxs in enumerate(indices)
        }

    @classmethod
    def from_pickle(self, data_path):
        dataset = pd.read_pickle(data_path)
        if not isinstance(dataset, ARDSDataset):
            raise ValueError('The pickle file you have specified is out-of-date. Please re-process your dataset and save the new pickled dataset.')
        # paranoia
        try:
            dataset.scaling_factors
        except AttributeError:
            dataset.derive_scaling_factors()
        return dataset

    def get_dataset(self):
        last_patient = None
        for fidx, filename in enumerate(self.raw_files):
            gen = read_processed_file(filename, filename.replace('.raw.npy', '.processed.npy'))
            patient_id = self._get_patient_id_from_file(filename)

            if patient_id != last_patient:
                batch_arr = []
                breath_arr = []
                seq_vent_bns = []
                batch_seq_hours = []

            last_patient = patient_id
            target = self._pathophysiology_target(patient_id)
            start_time = self._get_patient_start_time(patient_id)

            for bidx, breath in enumerate(gen):
                # cutoff breaths if they have too few points.
                if len(breath['flow']) < 21:
                    continue

                breath_time = self.get_abs_bs_dt(breath)
                if breath_time < start_time:
                    continue
                elif breath_time > start_time + pd.Timedelta(hours=24):
                    break

                flow = breath['flow']
                seq_hour = (breath_time - start_time).total_seconds() / 60 / 60
                seq_vent_bns.append(breath['vent_bn'])
                batch_arr, breath_arr, batch_seq_hours = self._unpadded_centered_processing(
                    flow, breath_arr, batch_arr, batch_seq_hours, seq_hour
                )

                if len(batch_arr) == self.n_sub_batches:
                    raw_data = np.array(batch_arr)
                    breath_window = raw_data.reshape((self.n_sub_batches, 1, self.seq_len))
                    self.all_sequences.append([patient_id, breath_window, target, batch_seq_hours])
                    batch_arr = []
                    seq_vent_bns = []
                    batch_seq_hours = []

                if len(batch_arr) > 0 and breath_arr == []:
                    batch_seq_hours.append(seq_hour)

    def get_abs_bs_dt(self, breath):
        if isinstance(breath['abs_bs'], bytes):
            breath['abs_bs'] = breath['abs_bs'].decode('utf-8')
        try:
            breath_time = pd.to_datetime(breath['abs_bs'], format='%Y-%m-%d %H-%M-%S.%f')
        except:
            breath_time = pd.to_datetime(breath['abs_bs'], format='%Y-%m-%d %H:%M:%S.%f')
        return breath_time

    def _pathophysiology_target(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        try:
            patient_row = patient_row.iloc[0]
        except:
            raise ValueError('Could not find patient {} in cohort file'.format(patient_id))
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        target = np.zeros(2)
        target[patho] = 1
        return target

    def _unpadded_centered_processing(self, flow, breath_arr, batch_arr, batch_seq_hours, seq_hour):
        if (len(flow) + len(breath_arr)) < self.seq_len:
            breath_arr.extend(flow)
        else:
            remaining = self.seq_len - len(breath_arr)
            breath_arr.extend(flow[:remaining])
            batch_arr.append(np.array(breath_arr))
            batch_seq_hours.append(seq_hour)
            breath_arr = []
        return batch_arr, breath_arr, batch_seq_hours

    def _get_scaling_factors_for_indices(self, indices):
        """
        Get mu and std for a specific set of indices
        """
        std_sum = 0
        mean_sum = 0
        obs_count = 0

        for idx in indices:
            obs = self.all_sequences[idx][1]
            obs_count += len(obs)
            mean_sum += obs.sum()
        mu = mean_sum / obs_count

        # calculate std
        for idx in indices:
            obs = self.all_sequences[idx][1]
            std_sum += ((obs - mu) ** 2).sum()
        std = np.sqrt(std_sum / obs_count)
        return mu, std

    def _pathophysiology_target(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        try:
            patient_row = patient_row.iloc[0]
        except:
            raise ValueError('Could not find patient {} in cohort file'.format(patient_id))
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        target = np.zeros(2)
        target[patho] = 1
        return target

    def _get_patient_start_time(self, patient_id):
        patient_row = self.cohort[self.cohort['patient_id'] == patient_id]
        patient_row = patient_row.iloc[0]
        patho = 1 if patient_row['Pathophysiology'] == 'ARDS' else 0
        if patho == 1:
            start_time = pd.to_datetime(patient_row['Date when Berlin criteria first met (m/dd/yyy)'])
        else:
            start_time = pd.to_datetime(patient_row['vent_start_time'])

        if start_time is pd.NaT:
            raise Exception('Could not find valid start time for {}'.format(patient_id))
        return start_time

    def __getitem__(self, index):
        seq = self.all_sequences[index]
        pt, data, target = seq
        try:
            mu, std = self.scaling_factors[None]
        except AttributeError:
            raise AttributeError('Scaling factors not found for dataset. You must derive them using the `derive_scaling_factors` function.')
        data = (data - mu) / std

        return pt, data, target

    def __len__(self):
        return len(self.all_sequences)

    def get_ground_truth_df(self):
        return self._get_all_sequence_ground_truth()

    def _get_all_sequence_ground_truth(self):
        rows = []
        for seq in self.all_sequences:
            patient, _, target = seq
            rows.append([patient, np.argmax(target, axis=0)])
        return pd.DataFrame(rows, columns=['patient', 'y'])

    def _get_patient_id_from_file(self, filename):
        pt_id = filename.split('/')[-2]
        # sanity check to see if patient
        match = re.search(r'(0\d{3}RPI\d{10})', filename)
        if match:
            return match.groups()[0]
        try:
            # id is from anonymous dataset
            float(pt_id)
            return pt_id
        except:
            raise ValueError('could not find patient id in file: {}'.format(filename))


class PVADataset(Dataset):
    def __init__(self, dataset_type, sequence_len):
        """
        Extract breath data and corresponding PVA annotations

        :param dataset_type: What set we are using (train/val/test)
        :param sequence_len: The length of the sequences we want to give to our LSTM
        """
        if dataset_type not in ['train', 'val', 'test']:
            raise Exception('dataset_type must be either "train", "val", or "test"')

        dataset_path = os.path.join(os.path.dirname(__file__), 'pva_dataset', dataset_type)
        self.record_set = glob(os.path.join(dataset_path, '*_data.pkl'))
        self.all_sequences = []
        self.sequence_len = sequence_len

    def process_dataset(self):
        """
        Extract all breaths in the dataset and pair a ground truth value
        with the breath information
        """
        for record in self.record_set:
            data = pd.read_pickle(record)
            gt = pd.read_pickle(record.replace('_data.pkl', '_gt.pkl'))
            patient = os.path.basename(record).split('_')[0]
            for i, b in enumerate(data):
                gt_row = gt.iloc[i]
                if gt_row.bn != b['rel_bn']:
                    raise Exception('something went wrong with gt parsing for record {}'.format(record))
                if gt_row.dta >= 1:
                    y = [0, 0, 1]
                elif gt_row.bsa >= 1:
                    y = [0, 1, 0]
                else:
                    y = [1, 0, 0]

                tensor = np.array([b['flow'], b['pressure']]).transpose()
                self.all_sequences.append([patient, tensor, np.array(y)])

        self.find_scaling_coefs()

    def find_scaling_coefs(self):
        """
        In order to conduct scaling you will need to find some scaling
        coefficients that are represented in our data. The time to find
        these coefficients is right after the data has been processed
        into a machine-usable format
        """
        # write function for finding the scaling coefficients
        raise Exception('you need to code me before things will run')

    def scale_breath(self, data):
        """
        Scale breath using any number of scaling techniques learned in
        this class. You can use standardization, max-min scaling, or
        anything else that you'd like to code
        """
        raise Exception('you need to code me before things will run')

    def pad_or_cut_breath(self, data):
        """
        For purposes of the simple LSTM that you are going to code you
        will need to have all your breaths be of uniform size. This means
        adding a padded value like 0 to a sequence to ensure a breath reaches
        desired length. It could also mean removing observations from a
        sequence if the data is longer than desired length
        """
        raise Exception('you need to code me before things will run')

    def __getitem__(self, idx):
        """
        get next sequence
        """
        pt, data, y = self.all_sequences[idx]
        data = self.scale_breath(data)
        data = self.pad_or_cut_breath(data)
        return data, y

    def __len__(self):
        return len(self.all_sequences)
