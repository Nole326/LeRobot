"""Independent action-trajectory replay verification and lossless synchronized BC observations."""
import json
import traceback
from pathlib import Path


from runtime import prepare_args, USD_SHA256


def main():
    args, AppLauncher = prepare_args(replay=True)
    if args is None:
        return
    import numpy as np
    from common import publish, sha256
    from evaluate import check_imported_sources
    out = args.out
    assert out.is_dir() and not (out/'replay.claim').exists()
    (out/'replay.claim').touch(exist_ok=False)
    assert json.loads((args.source/'result.json').read_text())['passed'], 'Expert validation must pass first'
    app = env = None
    videos = {}
    try:
        publish(out/'status.json', dict(status='running', phase='resource_admission'))
        import torch
        torch.set_num_threads(2)
        assert torch.cuda.device_count() == 1, 'Expose exactly one assigned GPU'
        assert torch.cuda.mem_get_info()[0] >= args.min_free_mib * 1024**2
        assert float((torch.ones(1, device='cuda')+1).cpu()[0]) == 2
        app = AppLauncher(args).app
        import gymnasium as gym
        import leisaac.tasks
        from isaaclab.managers import RecorderTermCfg, DatasetExportMode
        from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm
        from isaaclab.utils import configclass
        from isaaclab_tasks.utils import parse_env_cfg
        from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
        from leisaac.utils.robot_utils import (convert_leisaac_action_to_lerobot,
                                               is_so101_at_rest_pose)
        from joint_contract import JOINTS, motor_to_rad, rad_to_motor
        from rgb_video import RGBVideo

        def state(e):
            term = e.cfg.terminations.success
            robot = e.scene['robot']
            q = robot.data.joint_pos[:, ids]
            plate = e.scene[term.params['plate_cfg'].name].data.root_pos_w[0]
            objects = [e.scene[o.name] for o in term.params['oranges_cfg']]
            oranges = torch.stack([o.data.root_pos_w[0] for o in objects])
            relative = oranges-plate
            bounds = relative.new_tensor([term.params.get('x_range', (-.1,.1)),
                                           term.params.get('y_range', (-.1,.1)),
                                           term.params.get('height_range', (-.07,.07))])
            inside = ((relative>bounds[:,0]) & (relative<bounds[:,1])).all(-1)
            rest = bool(is_so101_at_rest_pose(robot.data.joint_pos, robot.data.joint_names)[0])
            success = bool(term.func(e, **term.params)[0])
            assert success == (bool(inside.all()) and rest)
            return dict(joint_motor=convert_leisaac_action_to_lerobot(q)[0].tolist(),
                        joint_degrees=(q[0]*180/torch.pi).cpu().tolist(),
                        joint_velocity=robot.data.joint_vel[0,ids].cpu().tolist(),
                        plate_xyz=plate.cpu().tolist(), oranges_xyz=oranges.cpu().tolist(),
                        oranges_quat=[o.data.root_quat_w[0].cpu().tolist() for o in objects],
                        oranges_velocity=[o.data.root_lin_vel_w[0].cpu().tolist() for o in objects],
                        inside=inside.cpu().tolist(), plate_count=int(inside.sum()), at_rest=rest,
                        native_success=success)

        class Capture(RecorderTerm):
            def record_pre_reset(self, env_ids):
                if getattr(self._env, 'replay_capture_enabled', False) and bool(self._env.reset_buf.any()):
                    self._env.replay_final = state(self._env)
                return None, None

        @configclass
        class Recorder(RecorderManagerBaseCfg):
            dataset_export_mode = DatasetExportMode.EXPORT_NONE
            export_in_record_pre_reset = False
            final_state = RecorderTermCfg(class_type=Capture)

        sources = check_imported_sources()
        cfg = parse_env_cfg('LeIsaac-SO101-PickOrange-v0', device=args.device, num_envs=1)
        cfg.use_teleop_device('so101leader')
        assert sha256(Path(cfg.scene.robot.spawn.usd_path)) == USD_SHA256, 'Robot calibration identity differs'
        cfg.seed = args.seeds[0]
        cfg.episode_length_s = 55
        cfg.sim.enable_scene_query_support = True
        cfg.scene.robot.spawn.rigid_props.disable_gravity = False
        cfg.recorders = Recorder()
        assert cfg.dynamic_reset_gripper_effort_limit
        env = gym.make('LeIsaac-SO101-PickOrange-v0', cfg=cfg).unwrapped
        robot = env.scene['robot']
        ids, names = robot.find_joints(JOINTS, preserve_order=True)
        assert names == JOINTS and abs(env.step_dt-1/60)<1e-7 and env.max_episode_length == 3300
        terms = [env.action_manager.get_term(k) for k in env.action_manager.active_terms]
        assert sum([list(t._joint_names) for t in terms], []) == JOINTS
        offsets = torch.cat([torch.as_tensor(t._offset, device=env.device).expand(1,t.action_dim) for t in terms], 1)
        scales = torch.cat([torch.as_tensor(t._scale, device=env.device).expand(1,t.action_dim) for t in terms], 1)
        base_out=out
        results=[]
        for seed in args.seeds:
            source=args.source/f'seed-{seed}'
            assert json.loads((source/'result.json').read_text())['success']
            out=base_out/f'seed-{seed}'
            out.mkdir(exist_ok=False)
            initial=torch.load(source/'initial-state.pt',map_location=env.device,weights_only=True)
            labels=np.load(source/'numeric-actions.npz',allow_pickle=False)
            actions=labels['action']
            original=labels['joints']
            env.replay_capture_enabled=False
            obs,_=env.reset_to(initial,env_ids=torch.tensor([0],device=env.device),seed=seed,is_relative=True)
            env.replay_capture_enabled=True;env.replay_final=None
            videos = {}
            for name in ('front', 'wrist'):
                videos[name] = RGBVideo(out/f'{name}.mkv')
            inputs=[];sent=[]
            publish(out/'status.json',dict(status='running'))
            with (out/'trace.jsonl').open('w') as trace:
                for step in range(3300):
                    if step%2==0:
                        action=actions[min(step//2,len(actions)-1)]
                        q=rad_to_motor(robot.data.joint_pos[0,ids].cpu().numpy())
                        for camera,video in videos.items():video.write(obs['policy'][camera][0].cpu().numpy())
                        inputs.append(q);sent.append(action.copy())
                        command=(torch.as_tensor(motor_to_rad(action),device=env.device).unsqueeze(0)-offsets)/scales
                    dynamic_reset_gripper_effort_limit_sim(env,'so101leader')
                    obs,_,term,trunc,_=env.step(command)
                    done=bool(term[0]|trunc[0]);current=env.replay_final if done else state(env)
                    trace.write(json.dumps(dict(step=step+1,diagnostics=current))+'\n')
                    if (step+1)%30==0 or done:
                        publish(out/'progress.json',dict(status='running',physics_steps=step+1,plate_count=current['plate_count'],at_rest=current['at_rest']))
                        with (out/'scalars.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,metrics=dict(plate_count=current['plate_count'],at_rest=int(current['at_rest']),success=int(current['native_success']))))+'\n')
                    if done:break
            video_info={camera:video.close() for camera,video in videos.items()}
            np.savez(out/'demonstration.npz',action=np.asarray(sent),joints=np.asarray(inputs),timestamp=np.arange(len(inputs))/30)
            n=min(len(original),len(inputs))
            differences=np.asarray([motor_to_rad(a)-motor_to_rad(b) for a,b in zip(inputs[:n],original[:n])])
            result=dict(seed=seed,success=current['native_success'],seconds=(step+1)/60,
                final=current,replay_rmse_rad=float(np.sqrt(np.mean(differences**2))),replay_max_error_rad=float(np.abs(differences).max()),
                videos=video_info,actor_inputs=['front_rgb','wrist_rgb','joint_state'],action_hz=30,observation_hz=30,
                source=str(source),training_eligible=bool(current['native_success']),privileged_actor_inputs=False)
            publish(out/'result.json',result)
            with (out/'scalars.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step+1,metrics=dict(
                    replay_rmse_rad=result['replay_rmse_rad'],
                    replay_max_error_rad=result['replay_max_error_rad'],
                    duration_seconds=result['seconds'],
                    training_eligible=int(result['training_eligible']),
                    rgb_frames=len(inputs))))+'\n')
            publish(out/'status.json',dict(status='completed'))
            results.append(dict(seed=seed,success=result['success'],seconds=result['seconds']))
            publish(base_out/'progress.json',dict(status='running',results=results))
        publish(base_out/'result.json',dict(results=results,passed=all(r['success'] for r in results)))
        publish(base_out/'status.json',dict(status='completed'))
        out=base_out
    except BaseException:
        publish(out/'error.json',dict(traceback=traceback.format_exc()))
        publish(out/'status.json',dict(status='failed'));raise
    finally:
        try:
            for video in videos.values():
                video.abort()
        finally:
            try:
                if env is not None:env.close()
            finally:
                if app is not None:app.close()

if __name__=='__main__':main()
