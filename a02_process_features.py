#%%
"""
Process image features from image preprocessed by various toolbox
    T1w: CAT12
    DTI: DiffusionKit
    ASL: TACA
"""
import os
import numpy as np
import pandas as pd

from datasets import mask
from datasets.subject import load_subjects
from datasets import utils

t1_mask_path = r'G:\006pd_DTI\02_mask\brainnetome\rBN_Atlas_246_1mm.nii'
t1_mask = mask.NiiMask(t1_mask_path)
subjects = load_subjects(r'G:\006pd_DTI\subject_info.csv')

def asl_rename(asl):
    asl_dir = asl.directory
    files = os.listdir(asl_dir)
    # be caution
    for f in files:
        if '_1.nii' in f:
            asl.rename(f, 'asl.nii', use_mark=False)
        elif '_2.nii' in f:
            asl.rename(f, 'pd.nii', use_mark=False)

def asl_corregister(observation):
    aladin_path = r'G:/006pd_DTI/pd_vim/utils/reg_aladin.exe'
    resample_path = r'G:/006pd_DTI/pd_vim/utils/reg_resample.exe'
    mni_ref_path = r'G:/006pd_DTI/02_mask/mni_temp/Template_T1_IXI555_MNI152_GS.nii'

    asl = observation.asl
    t1 = observation.t1

    # register T1 to MNI standard space
    t1_path = t1.build_path('t1.nii',use_mark=False)
    t1_mni_path = t1.build_path('t1_mni.nii',use_mark=False)
    t1_aff_path = t1.build_path('t1_mni_affine.txt',use_mark=False)

    utils.reg_aladin(aladin_path, mni_ref_path, t1_path, t1_mni_path, t1_aff_path)

    # register pd to T1 space
    pd_path = asl.build_path('pd.nii', use_mark=False)
    pd_t1_path = asl.build_path('pd_t1.nii', use_mark=False)
    pd_aff_path = asl.build_path('pd_t1_affine.txt', use_mark=False)

    utils.reg_aladin(aladin_path, t1_path, pd_path, pd_t1_path, pd_aff_path)

    # resample asl to pd-T1 space using pd-T1 affine
    asl_path = asl.build_path('asl.nii', use_mark=False)
    asl_t1_path = asl.build_path('asl_t1.nii', use_mark=False)

    utils.reg_resample(resample_path, pd_path, asl_path, asl_t1_path, pd_aff_path)

    # resample pd-T1/asl-T1 to T1-MNI space using T1-MNI affine
    pd_mni_path = asl.build_path('pd_mni.nii', use_mark=False)
    utils.reg_resample(resample_path, t1_mni_path, pd_t1_path, pd_mni_path, t1_aff_path)

    asl_mni_path = asl.build_path('asl_mni.nii', use_mark=False)
    utils.reg_resample(resample_path, t1_mni_path, asl_t1_path, asl_mni_path, t1_aff_path)

for subject in subjects:
    for observation in subject.get_all_observation():
        t1 = observation.t1
        dti = observation.dti
        asl = observation.asl
        
        # T1 process
        ## create smoothed segmented grey matter image
        t1.create_smoothed_nii('mri/mwp1{}.nii',
                                'mri/smwp1_{}.nii',
                                fwhm=4)
        ## calculate ROI GMV using smoothed image
        t1.create_roi_volume_csv(t1_mask, 'mri/smwp1_{}.nii',
                                out_csv_filename='label/roi_gmv_{}.csv')

        t1.create_roi_ct_from_cat(cortical_id_path=r'G:\006pd_DTI\02_mask\brainnetome\cortical_id_new.csv')
        
        # DTI process
        dti.create_roi_volume_csv(image_name='dti_FA.nii.gz',
                                  out_csv_filename='roi_fa.csv')
        dti.create_roi_volume_csv(image_name='dti_MD.nii.gz',
                                  out_csv_filename='roi_md.csv')

        # Flip R-side observation features to L-side
        # Be caution!
        if observation.lesion_side == 'right':
            def exchange_roi_value(ori_path, backup_path):
                # backup original file only once
                if not os.path.exists(backup_path):
                    os.rename(ori_path, backup_path)
                df = pd.read_csv(backup_path, index_col='ID')
                # exchange roi id by pair (1<->2) (3<->4)
                index = []
                for i in range(1, 247):
                    if i%2 == 1:
                        j = i+1
                    else:
                        j = i-1
                    index.append(j)
                df = df.reindex(index)
                new_index = [k for k in range(1,247)]
                df.index = new_index
                df.to_csv(ori_path)
            # exchange roi_gmv
            ori_path = t1.build_path('label/roi_gmv_{}.csv')
            backup_path = t1.build_path('label/roi_gmv_{}_backup.csv')
            exchange_roi_value(ori_path, backup_path)
            # exchange roi_ct
            ori_path = t1.build_path('label/roi_ct_{}.csv')
            backup_path = t1.build_path('label/roi_ct_{}_backup.csv')
            exchange_roi_value(ori_path, backup_path)
            # exchange roi_cbf
            try:
                ori_path = asl.build_path('roi_mean_cbf.csv', use_mark=False)
                backup_path = asl.build_path('roi_mean_cbf_backup.csv', use_mark=False)
                exchange_roi_value(ori_path, backup_path)
            except FileNotFoundError as e:
                print('NO ASL file for {}:{}'.format(subject.name, observation.name))
            # exchange roi_fa
            ori_path = dti.build_path('roi_fa.csv', use_mark=False)
            backup_path = dti.build_path('roi_fa_backup.csv', use_mark=False)
            exchange_roi_value(ori_path, backup_path)
            # exchange roi_md
            ori_path = dti.build_path('roi_md.csv', use_mark=False)
            backup_path = dti.build_path('roi_md_backup.csv', use_mark=False)
            exchange_roi_value(ori_path, backup_path)

            def exchange_matrix(ori_path, backup_path, fmt='%10f'):
                if not os.path.exists(backup_path):
                    os.rename(ori_path, backup_path)
                array = np.loadtxt(backup_path, dtype=np.float32)
                for i in range(0, 246, 2):
                    array[:,[i, i+1]] = array[:,[i+1, i]]
                    array[[i, i+1], :] = array[[i+1, i], :]
                np.savetxt(ori_path, array, fmt=fmt)
            # exchange network_num
            ori_path = dti.build_path('network_num.txt', use_mark=False)
            backup_path = dti.build_path('network_num_backup.txt', use_mark=False)
            exchange_matrix(ori_path, backup_path, fmt='%d')
            # exchange network_fa
            ori_path = dti.build_path('network_fa.txt', use_mark=False)
            backup_path = dti.build_path('network_fa_backup.txt', use_mark=False)
            exchange_matrix(ori_path, backup_path)
            # exchange network_md
            ori_path = dti.build_path('network_md.txt', use_mark=False)
            backup_path = dti.build_path('network_md_backup.txt', use_mark=False)
            exchange_matrix(ori_path, backup_path)