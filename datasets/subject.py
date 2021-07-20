"""
Subject management module
"""
import ast
import os
import csv

import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd

from . import modal

class Observation(object):
    def __init__(self, name, args):
        self.name = name

        self.directory = args['dir']
        self.lesion_side = args['lesion_side']

        self.hand_tremor = args['handtremor']
        self.crst_a_total = args['CRSTA_total']
        self.crst_b_total = args['CRST b_total']
        self.crst_c_total = args['CRST C']
        self.crst_total = args['CRST TOTAL']
        self.args = args

        self.t1 = modal.T1(os.path.join(self.directory, 'T1_post'))
        self.dti = modal.Dti(os.path.join(self.directory, 'DTI_post'))
        self.asl = modal.Asl(os.path.join(self.directory, 'ASL'))

    def get_lesion_side(self):
        return self.lesion_side

class Subject(object):
    """
    A subect that contains multiple observation accross time,
    each observation contains multimodal images including:
        1. T1
        2. DTI
        3. BOLD
        4. ASL
        5. ESWAN
    Perform operations including read, write and feature extraction
    subject dir structure example (load observation from center_info.csv):
        subject01_observation01
            |---t1
            |---dti
            |---asl
        subject01_observation02
            |---t1
            |---dti
            |---asl
    Attributes:
        name: string, name for the subject.
        observations: list of Observation, holds observations for this subject.
    Functions:
        load_personal_info: load personal info like age, sex, etc.
        load_observation: load all observations for this subject.
        get_label: get subject label.
        get_personal_info: return a dict of personal info.
        get_observation: get observation with certain name.
    """
    def __init__(self, name,
                 observations_dict):
        """
        Args:
            name: string, name for the subject.
            observations_dict: dict of observations, {name1:args1, name2:args2}
        """
        self.name = name
        self.load_observation(observations_dict)

        self.load_personal_info()

    def load_personal_info(self):
        observation = self.get_observation('base')
        self.lesion_side = observation.lesion_side

    def load_observation(self, observations_dict):
        self.observations = []
        for observation_name, observation_args in observations_dict.items():
            observation = Observation(observation_name, observation_args)
            self.observations.append(observation)

    def get_label(self):
        if not self.label:
            self.load_personal_info()
        return self.label

    def get_personal_info(self):
        pass

    def get_observation(self, observation_name):
        for observation in self.observations:
            if observation.name == observation_name:
                return observation
    
    def get_all_observation(self):
        return self.observations

def load_subjects(info_path):
    info_df = pd.read_csv(info_path, index_col=['subject', 'observation'])
    subject_names = set(info_df.index.get_level_values(0))
    subjects = []
    for subject_name in subject_names:
        observations = info_df.loc[subject_name,:]
        observations_dict = {}
        for index, row in observations.iterrows():
            observations_dict[str(index)] = row
        subject = Subject(subject_name, observations_dict)
        subjects.append(subject)
    return subjects