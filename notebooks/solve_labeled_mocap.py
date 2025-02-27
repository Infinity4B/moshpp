eval_folder_name='evaluation_mocaps_cgm2_43'
target_ds_names = ['0221_afternoon_all']

import os.path as osp
from glob import glob

import numpy as np
from loguru import logger
import sys
import os
import gc

from moshpp.mosh_head import MoSh
from moshpp.mosh_head import run_moshpp_once

def mosh_manual(
        mocap_fnames: list,
        mosh_cfg: dict = None,
        **kwargs):
    if mosh_cfg is None: mosh_cfg = {}

    for mocap_fname in mocap_fnames:
        mosh_job = mosh_cfg.copy()
        mosh_job.update({
            'mocap.fname': mocap_fname,
        })

        run_moshpp_once(mosh_job)
        torch.cuda.empty_cache()
        gc.collect()

work_base_dir = '/home/pc/Documents/Code/mymoshpp/'
support_base_dir = osp.join(work_base_dir, 'support_files')

mocap_base_dir = osp.join(support_base_dir, eval_folder_name)

result_base_dir = osp.join(work_base_dir, 'mosh_results')

data_base_dir=osp.join(work_base_dir,'data_results')

from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch

# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(comp_device)

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools.vis_tools import show_image
from body_visualizer.tools.vis_tools import imagearray2file

imw, imh = 800, 800
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
from human_body_prior.body_model.body_model import BodyModel

GENERATE_VIDEO=False

for ds_name in target_ds_names:
    mocap_fnames = glob(osp.join(mocap_base_dir, ds_name,  '*.c3d'))

    logger.info(f'#mocaps found for {ds_name}: {len(mocap_fnames)}')

    mosh_manual(
        mocap_fnames,
        mosh_cfg={
            'moshpp.verbosity': 1, # set to 2 to visulaize the process in meshviewer
            'dirs.work_base_dir': result_base_dir,
            'dirs.support_base_dir': support_base_dir,
            # 'mocap.end_fidx': 100,  # comment in real runs
        },
        parallel_cfg={
            'pool_size': 1,
            'max_num_jobs': 1,
            'randomly_run_jobs': True,
        },
    )
    for mocap_fname in mocap_fnames:
        real_mocap_fname = mocap_fname.split('/')[-1].replace('.c3d','_stageii')
        mosh_stageii_pkl_fname = osp.join(result_base_dir, eval_folder_name, ds_name, real_mocap_fname+'.pkl')
        mosh_stageii_output_fname = osp.join(data_base_dir, eval_folder_name, ds_name, real_mocap_fname+'.npz')

        logger.info(f'transforming {mosh_stageii_pkl_fname} to {mosh_stageii_output_fname}')
        mosh_result = MoSh.load_as_amass_npz(mosh_stageii_pkl_fname,stageii_npz_fname=mosh_stageii_output_fname)
        # print({k:v if isinstance(v, str) or isinstance(v,float) or isinstance(v,int) else v.shape for k,v in mosh_result.items() if not isinstance(v, list) and not isinstance(v,dict)})
        # print(mosh_result.keys())

        time_length = len(mosh_result['trans'])
        mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)

        subject_gender = mosh_result['gender']
        surface_model_type = mosh_result['surface_model_type']
        logger.info(f'subject_gender: {subject_gender}, surface_model_type: {surface_model_type}, time_length: {time_length}')

        if GENERATE_VIDEO:
            amass_npz_fname = mosh_stageii_output_fname # the path to body data
            bdata = np.load(amass_npz_fname,allow_pickle=True)

            num_betas = 16 # number of body parameters
            num_dmpls = 8 # number of DMPL parameters

            # you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
            subject_gender = bdata['gender']

            logger.info('Data keys available:%s'%list(bdata.keys()))

            time_length = len(bdata['trans'])

            body_parms = {
                'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
                'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
                'pose_hand': torch.Tensor(bdata['poses'][:, 66:66+90]).to(comp_device), # controls the finger articulation
                # 'pose_hand': torch.zeros((time_length, 90)).to(comp_device), # controls the finger articulation
                'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
                'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
                'dmpls': torch.zeros((time_length,8)).to(comp_device) # controls soft tissue dynamics
            }


            logger.info('Body parameter vector shapes: {}'.format(' '.join(['{}: {}'.format(k,v.shape) for k,v in body_parms.items()])))
            logger.info('time_length = {}'.format(time_length))

            bm_smplx_fname = osp.join(support_base_dir, 'smplx/{}/model.npz'.format(subject_gender))

            bm = BodyModel(bm_fname=bm_smplx_fname, num_betas=num_betas).to(comp_device)

            faces = c2c(bm.f)
            num_verts = bm.init_v_template.shape[1]

            logger.info({k:v.shape for k,v in body_parms.items() if k in ['pose_body', 'pose_hand', 'betas']})
            body = bm(**{k:v.to(comp_device) for k,v in body_parms.items() if k in ['pose_body', 'pose_hand','betas']})
            # body_mesh_wfingers = trimesh.Trimesh(vertices=c2c(body.v[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (num_verts, 1)))
            # mv.set_static_meshes([body_mesh_wfingers])
            # body_image_wfingers = mv.render(render_wireframe=False)
            # show_image(body_image_wfingers)

            image_arr = []
            for fId in range(time_length):
                body_mesh_wo_dmpl = trimesh.Trimesh(vertices=c2c(body.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
                meshes=[body_mesh_wo_dmpl]
                mv.set_static_meshes(meshes)
                body_image = mv.render(render_wireframe=False)
                image_arr.append(body_image)

            image_arr = np.array(image_arr).reshape([1,1,-1, imw, imh, 3])
            output_mp4_name=osp.join(data_base_dir,eval_folder_name,ds_name,real_mocap_fname+'.mp4')
            logger.info(f'Generating mp4 file to {output_mp4_name}')
            out_images=imagearray2file(image_arr, outpath=output_mp4_name, fps=60)
